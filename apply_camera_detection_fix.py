"""Add a second ArUco threshold pass when the primary detector misses ID 0."""
import argparse
from pathlib import Path
from datetime import datetime
import shutil
MARKER = '# CLOSE_MARKER_FALLBACK_V1'
def update(source):
    if MARKER in source: raise ValueError('Already installed.')
    old="""        h = MARKER_SIZE / 2"""
    new="""        # Independent pass: avoid interactions between threshold-window candidates.
        # Keep the primary detector unchanged and use identical corner refinement.
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 3
        self.close_detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)
        # CLOSE_MARKER_FALLBACK_V1
        h = MARKER_SIZE / 2"""
    old2="""        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))"""
    new2="""        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or not np.any(ids == 0):
            corners, ids, _ = self.close_detector.detectMarkers(gray)
        # Both passes use original-resolution pixels and the same PnP/quality gates."""
    for a,b in [(old,new),(old2,new2)]:
        if source.count(a)!=1: raise ValueError('Unexpected test_ekf.py structure; no change made.')
        source=source.replace(a,b,1)
    compile(source,'test_ekf.py','exec')
    return source
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true');args=parser.parse_args()
    p=Path(__file__).resolve().parent/'test_ekf.py'
    try:
        result=update(p.read_text(encoding='utf-8-sig'))
        if args.check: print('Compatible; no changes made.')
        else:
            backup=p.with_name('test_ekf_before_camera_fix_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.py')
            shutil.copy2(p,backup)
            tmp=p.with_suffix('.camera_tmp');tmp.write_text(result,encoding='utf-8');tmp.replace(p)
            print('Installed. Backup:',backup)
    except (OSError,ValueError,SyntaxError) as exc:parser.exit(1,str(exc)+'\n')
