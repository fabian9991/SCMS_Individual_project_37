"""Fixed-reference LQR recovery tests using the existing navigation components.

Place beside navigate_all_goals.py, test_ekf.py, lqr_diagnostics.py and
test_wheel_response.py from the diagnostics package. No mission source is edited.
Run: python test_lqr_recovery.py --seed 37 --viewer
Truth is used only for scoring. Test fixture poses are declared, not read back.
"""
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
import argparse
import csv
from datetime import datetime
import json
import time
import traceback
import numpy as np
import test_ekf as ekf
from lqr_diagnostics import ControllerLog, write_configuration
from test_wheel_response import navigation_module

ROOT = Path(__file__).resolve().parent
ORIGIN = np.array([-4., 0.])
REFERENCE_HEADING = np.pi
STRAIGHT = np.array([[0., 0.], [1.5, 0.]])
CORNER = np.array([[0., 0.], [1., 0.], [1., -.4]])
# x forward and y left in the declared reference frame. At heading pi,
# positive lateral offset places the robot south (negative world y).
CASES = [
    dict(name='on_path', lateral=0., heading_deg=0., mode='TRACK', route=STRAIGHT),
    dict(name='sideways_left', lateral=.05, heading_deg=0., mode='TRACK', route=STRAIGHT),
    dict(name='sideways_right', lateral=-.05, heading_deg=0., mode='TRACK', route=STRAIGHT),
    dict(name='heading_left', lateral=0., heading_deg=10., mode='TRACK', route=STRAIGHT),
    dict(name='heading_right', lateral=0., heading_deg=-10., mode='TRACK', route=STRAIGHT),
    dict(name='alignment_transition', lateral=0., heading_deg=30., mode='ALIGN', route=STRAIGHT),
    dict(name='corner', lateral=0., heading_deg=0., mode='TRACK', route=CORNER),
]
MAX_CASE_TIME = 60.
ENDPOINT_TOL = .02  # Existing Tracker default; estimated position only.
RECOVERY_Y = .01   # Diagnostic band, not an arrival or pass/fail rule.
RECOVERY_HEADING = np.deg2rad(2.)
RECOVERY_HOLD = 1.


def world_geometry(case):
    c, s = np.cos(REFERENCE_HEADING), np.sin(REFERENCE_HEADING)
    rotation = np.array([[c, -s], [s, c]])
    route = ORIGIN + np.asarray(case['route']) @ rotation.T
    xy = ORIGIN + rotation @ np.array([0., case['lateral']])
    prior = np.array([*xy, ekf.wrap_angle(REFERENCE_HEADING + np.deg2rad(case['heading_deg']))])
    return prior, route


def errors(pose, a, b):
    direction = np.arctan2(b[1]-a[1], b[0]-a[0])
    normal = np.array([-np.sin(direction), np.cos(direction)])
    return float((pose[:2]-a) @ normal), float(ekf.wrap_angle(pose[2]-direction))


def recovery_entry(samples, prefix, segment):
    """First band entry sustained for one second of uninterrupted TRACK samples.
    Return None if no such interval was observed. Later departures are possible.
    """
    entry = None
    for r in samples:
        inside = (r['segment'] == segment and r['status'] == 'TRACK'
                  and not r['stop'] and abs(r[prefix+'_ey']) <= RECOVERY_Y
                  and abs(r[prefix+'_eth']) <= RECOVERY_HEADING)
        if not inside:
            entry = None
        elif entry is None:
            entry = r['time_s']
        elif r['time_s']-entry >= RECOVERY_HOLD:
            return float(entry)
    return None


def result_rows(case, seed, samples, route, outcome, counts, initial, arrival):
    """One summary row per reference segment; errors scored at control times."""
    rows = []
    for seg in range(len(route)-1):
        all_segment = [r for r in samples if r['segment'] == seg]
        track = [r for r in all_segment if r['status'] == 'TRACK' and not r['stop']]
        row = dict(case=case['name'], seed=seed, segment=seg, outcome=outcome,
                   track_samples=len(track), camera_accepted=counts['accepted'],
                   actual_initial_ey_m=initial['actual_ey'],
                   actual_initial_heading_deg=np.degrees(initial['actual_eth']),
                   initial_localisation_error_m=initial['localisation_error'],
                   first_track_time_s=track[0]['time_s'] if track else None,
                   alignment_duration_s=sum(r['dt'] for r in all_segment if r['status']=='ALIGN'),
                   lidar_stop_samples=sum(r['status']=='LIDAR STOP' for r in all_segment),
                   actuator_limit_samples=sum(r['actuator_limit'] for r in all_segment),
                   endpoint_arrival_time_s=arrival.get('time_s'),
                   estimated_endpoint_distance_m=arrival.get('estimated_distance_m'),
                   actual_endpoint_distance_m=arrival.get('actual_distance_m'))
        for prefix in ('estimated', 'actual'):
            for name in ('ey', 'eth'):
                v = np.array([r[prefix+'_'+name] for r in track])
                key = prefix + ('_cross_track' if name=='ey' else '_heading')
                scale = 1. if name=='ey' else 180/np.pi
                unit = '_m' if name=='ey' else '_deg'
                row[key+'_rms'+unit] = float(np.sqrt(np.mean(v*v))*scale) if len(v) else None
                row[key+'_max_abs'+unit] = float(np.max(np.abs(v))*scale) if len(v) else None
            row[prefix+'_recovery_entry_time_s'] = recovery_entry(samples, prefix, seg)
        rows.append(row)
    return rows


def write_summary(path, records):
    if not records:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run_case(nav, mujoco, model, data, ids, camera, lidar, viewer, output, case, seed):
    directory = output/case['name']
    directory.mkdir()
    declared_prior, route = world_geometry(case)
    with viewer.lock() if viewer is not None else nullcontext():
        mujoco.mj_resetData(model, data)
        address = model.jnt_qposadr[ids['free_joint']]
        # Physical fixture setup from declared constants, performed once per case.
        data.qpos[address:address+2] = declared_prior[:2]
        half = declared_prior[2]/2
        data.qpos[address+3:address+7] = [np.cos(half), 0., 0., np.sin(half)]
        for _ in range(round(1./model.opt.timestep)):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        # Estimator uses the declared prior, never evaluation_pose().
        pose, P = declared_prior.copy(), nav.START_P.copy()
        previous = data.qpos[ids['wheel_addresses']].copy()
        control_encoders = previous.copy()
        start = last_control = data.time
        truth0 = nav.evaluation_pose(data, ids['base'])  # Scoring only.
    tracker, servo = nav.Tracker(), nav.WheelServo()
    tracker.route, tracker.mode = route.copy(), case['mode']
    context = SimpleNamespace(tracker=tracker, events=[], mode=tracker.mode,
                              route_id=1, segment=0, plan_status='FIXED TEST ROUTE')
    rng = np.random.default_rng(seed)
    aey, aeth = errors(truth0, route[0], route[1])
    initial = dict(actual_ey=aey, actual_eth=aeth,
                   localisation_error=float(np.linalg.norm(pose[:2]-truth0[:2])))
    write_configuration(directory/'case_config.json', dict(
        case=case, seed=seed, declared_prior=declared_prior, route_world=route,
        initial_scoring=initial, start_covariance=P,
        note='Known physical offset; no deliberate prior error. No truth feedback.'), [])
    log = ControllerLog(directory, nav.WHEEL_RADIUS, nav.WHEEL_SEPARATION)
    log.event(0., 'fixed_route', route=route, start_mode=case['mode'], declared_prior=declared_prior)
    samples, counts, arrival = [], dict(attempts=0, detected=0, accepted=0, rejected=0, far=0), {}
    next_camera=next_control=next_print=0.
    done_since = None
    outcome = 'TIMEOUT'
    failure = None
    try:
        while data.time-start < MAX_CASE_TIME:
            if viewer is not None and not viewer.is_running():
                outcome='VIEWER CLOSED'
                break
            wall=time.perf_counter()
            with viewer.lock() if viewer is not None else nullcontext():
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                enc=data.qpos[ids['wheel_addresses']].copy()
                increments=nav.noisy_wheel_increments(enc-previous, rng)
                pose,P=nav.predict_ekf(pose,P,increments,model.opt.timestep,
                                     wheel_noise_coeff=nav.ENCODER_NOISE_M_PER_SQRT_M)
                previous=enc
                t=data.time-start
                if t>=next_camera:
                    next_camera=t+nav.DT_CAMERA
                    counts['attempts']+=1
                    z=camera.read(data)
                    if z is not None:
                        counts['detected']+=1
                        if z[0]<=nav.CAMERA_USE_RANGE:
                            pose,P,nis,ok,_=nav.correct_ekf(pose,P,z)
                            counts['accepted' if ok else 'rejected']+=1
                            log.event(t,'camera_correction',accepted=ok,nis=nis,range_m=z[0])
                        else:
                            counts['far']+=1
                if t>=next_control:
                    elapsed=data.time-last_control
                    rates=(enc-control_encoders)/elapsed
                    control_encoders=enc.copy()
                    last_control=data.time
                    next_control=t+nav.DT_CONTROL
                    segment_before=tracker.segment
                    a,b=route[segment_before:segment_before+2]
                    v,omega,_,_=tracker.command(pose, ENDPOINT_TOL)
                    if tracker.segment != segment_before:
                        log.event(t,'segment_transition',previous=segment_before,current=tracker.segment)
                    ranges=lidar.read(data)
                    blocked,clearance=nav.blocked_by_scan(ranges,v,omega)
                    stop=tracker.mode in ('DONE','OVERSHOOT')
                    status=tracker.mode
                    if blocked and not stop:
                        status='LIDAR STOP'
                        stop=True
                    desired=nav.wheel_targets(v,omega)
                    commands=servo.command(desired,rates,elapsed,stop)
                    data.ctrl[ids['motors']]=commands
                    # Evaluation block. No truth or error metrics feed control.
                    truth=nav.evaluation_pose(data,ids['base'])
                    context.mode=tracker.mode
                    log.record(t,elapsed,context,servo,pose,truth,v,omega,desired,rates,
                               commands,stop,status,clearance)
                    ey,eth=errors(pose,a,b)
                    ay,ath=errors(truth,a,b)
                    sample=dict(time_s=t,dt=elapsed,segment=segment_before,status=status,
                                stop=bool(stop),estimated_ey=ey,estimated_eth=eth,
                                actual_ey=ay,actual_eth=ath,
                                actuator_limit=bool(np.any(np.abs(servo.diagnostics['raw'])>=7.88-1e-12)))
                    samples.append(sample)
                    if status=='DONE':
                        if done_since is None:
                            done_since=t
                            arrival=dict(time_s=t,estimated_distance_m=float(np.linalg.norm(pose[:2]-route[-1])),
                                         actual_distance_m=float(np.linalg.norm(truth[:2]-route[-1])))
                        if t-done_since>=2.:
                            outcome='ESTIMATED ENDPOINT REACHED'
                            break
                    if status in ('OVERSHOOT','LIDAR STOP'):
                        outcome=status  # A stopped/unfinished test is retained, not discarded.
                        break
                    if t>=next_print:
                        next_print=t+2.
                        print(f"{case['name']:22s} {t:5.1f}s {status:10s} "
                              f"estimated ey={100*ey:+.2f} cm; actual ey={100*ay:+.2f} cm")
            if viewer is not None:
                viewer.sync()
                delay=model.opt.timestep-(time.perf_counter()-wall)
                if delay>0:time.sleep(delay)
    except KeyboardInterrupt:
        outcome='INTERRUPTED'
    except Exception as exc:
        failure=exc
        outcome=f'ERROR: {type(exc).__name__}: {exc}'
        (directory/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
    finally:
        data.ctrl[ids['motors']]=0
        log.event(data.time-start,'run_end',outcome=outcome,camera_counts=counts)
        log.close()
        summary=result_rows(case,seed,samples,route,outcome,counts,initial,arrival)
        write_summary(directory/'tracking_summary.csv',summary)
        (directory/'summary.txt').write_text(outcome+'\n'+json.dumps(dict(
            initial=initial,arrival=arrival,camera_counts=counts),indent=2)+'\n',encoding='utf-8')
    if failure is not None:raise failure
    return summary,outcome


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=37)
    parser.add_argument('--case',choices=['all']+[c['name'] for c in CASES],default='all')
    parser.add_argument('--viewer',action='store_true')
    args=parser.parse_args()
    import mujoco
    nav=navigation_module()
    if not nav.ENCODER_NOISE_ENABLED or not nav.USE_WHEEL_TRAVEL_Q or nav.SLIP_PLATE_ENABLED:
        raise RuntimeError('Use the fixed baseline: encoder noise on, wheel-travel Q on, slip plate off.')
    scene=ROOT/'models/robotis_tb3/scene_assignment_plate.xml'
    model=mujoco.MjModel.from_xml_path(str(scene))
    data=mujoco.MjData(model)
    def required(kind,name):
        i=mujoco.mj_name2id(model,kind,name)
        if i<0:raise ValueError(f'Missing model object: {name}')
        return i
    kinds=mujoco.mjtObj
    base=required(kinds.mjOBJ_BODY,'base')
    joints=[required(kinds.mjOBJ_JOINT,n) for n in ('wheel_left','wheel_right')]
    ids=dict(base=base,free_joint=required(kinds.mjOBJ_JOINT,'base_joint'),
             wheel_addresses=model.jnt_qposadr[joints],
             motors=[required(kinds.mjOBJ_ACTUATOR,n) for n in ('wheel_left','wheel_right')])
    floor=required(kinds.mjOBJ_GEOM,'floor');plate=required(kinds.mjOBJ_GEOM,'slip_plate')
    model.geom_friction[plate]=model.geom_friction[floor]
    lidar=nav.Lidar(model,base,required(kinds.mjOBJ_SITE,'lidar_site'))
    camera=nav.MarkerCamera(model)
    output=ROOT/'results'/('lqr_recovery_seed_'+str(args.seed)+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True)
    selected=[c for c in CASES if args.case=='all' or c['name']==args.case]
    write_configuration(output/'run_config.json',dict(
        seed=args.seed,cases=[c['name'] for c in selected],encoder_noise_enabled=True,
        encoder_noise_coefficient=nav.ENCODER_NOISE_M_PER_SQRT_M,use_wheel_travel_Q=True,
        slip_plate_enabled=False,camera_R=ekf.R_CAMERA,camera_period_s=nav.DT_CAMERA,
        camera_max_range_m=nav.CAMERA_USE_RANGE,control_dt_s=nav.DT_CONTROL,
        endpoint_estimated_tolerance_m=ENDPOINT_TOL,max_case_time_s=MAX_CASE_TIME,
        recovery_cross_track_band_m=RECOVERY_Y,recovery_heading_band_deg=2.,recovery_hold_s=RECOVERY_HOLD,
        gains=[dict(v=float(v),K=nav.lqr_gain(v)[0]) for v in nav.Tracker().speeds],
        reference_heading_rad=REFERENCE_HEADING,fixture_origin_m=ORIGIN,
        mujoco_version=mujoco.__version__,numpy_version=np.__version__,
        note='Fixed route/controller test; LiDAR stop active; no planner and no truth feedback.'),
        sources=[Path(__file__),Path(nav.__file__),Path(ekf.__file__),ROOT/'lqr_diagnostics.py',
                 *sorted(scene.parent.glob('*.xml'))])
    print('Output folder:',output)
    viewer=None
    summaries=[]
    try:
        if args.viewer:
            import mujoco.viewer
            viewer=mujoco.viewer.launch_passive(model,data)
        for case in selected:
            summary,outcome=run_case(nav,mujoco,model,data,ids,camera,lidar,viewer,output,case,args.seed)
            summaries.extend(summary)
            write_summary(output/'all_cases_summary.csv',summaries)
            print(case['name']+': '+outcome)
            if outcome in ('VIEWER CLOSED','INTERRUPTED'):break
    finally:
        data.ctrl[ids['motors']]=0
        if viewer is not None:viewer.close()
        camera.close()
        print('Results:',output)


if __name__=='__main__':
    main()
