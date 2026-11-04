"""LiDAR occupancy mapping and goal/frontier planning. NumPy + standard library.
Inputs are estimated planar pose and robot-relative scan ranges, never XML
obstacle locations or simulator body poses. Fixed bounds are workspace limits,
not a supplied obstacle map. Static-world mapper; not SLAM or dynamic tracking.
"""
import heapq
import numpy as np

class LidarPlanner:
    def __init__(self, resolution=.08, radius=.27, margin=.05):
        self.res = resolution
        self.origin = np.array([-10., -4.])
        self.shape = (76, 139)  # y,x; extends to approximately (1.1,2.1)
        yy,xx = np.indices(self.shape)
        self.xy = self.origin + np.stack([xx,yy],axis=-1)*self.res
        self.odds = np.zeros(self.shape)
        self.visits = np.zeros(self.shape)
        self.padding = radius+margin+self.res/np.sqrt(2)
        self.unknown_padding = radius+self.res/2
        self.safe = np.zeros(self.shape,dtype=bool)
        self.status = 'UNMAPPED'
        self.plans = 0

    def cell(self, xy):
        c=np.rint((np.asarray(xy)-self.origin)/self.res).astype(int)
        return int(c[1]),int(c[0])

    def inside(self,c):
        return 0<=c[0]<self.shape[0] and 0<=c[1]<self.shape[1]

    def dilate(self, mask, radius):
        out=np.zeros_like(mask)
        n=int(np.ceil(radius/self.res));h,w=mask.shape
        for dy in range(-n,n+1):
            for dx in range(-n,n+1):
                if np.hypot(dx,dy)*self.res>radius: continue
                ya,yb=max(0,dy),min(h,h+dy);xa,xb=max(0,dx),min(w,w+dx)
                out[ya:yb,xa:xb] |= mask[ya-dy:yb-dy,xa-dx:xb-dx]
        return out

    def update(self,pose,ranges):
        ranges=np.asarray(ranges)
        if ranges.shape!=(360,): raise ValueError('Expected 360 LiDAR ranges.')
        if not np.any(np.isfinite(ranges)&(ranges>0)): return
        heading=pose[2]
        origin=pose[:2]-.049*np.array([np.cos(heading),np.sin(heading)])
        delta=self.xy-origin
        distance=np.linalg.norm(delta,axis=-1)
        angles=(np.arctan2(delta[...,1],delta[...,0])-heading)%(2*np.pi)
        beam=np.rint(np.degrees(angles)).astype(int)%360
        # Conservatively use the shortest of adjacent beams at cell centres.
        limits=np.minimum.reduce([ranges,np.roll(ranges,1),np.roll(ranges,-1)])
        limits=np.where(np.isposinf(limits),4.,limits)
        free=(distance<np.minimum(limits[beam]-.04,4.))&(limits[beam]>0)
        self.odds[free]=np.maximum(-4.,self.odds[free]-.5)
        valid=np.isfinite(ranges)&(ranges>0)&(ranges<=4.)
        aa=np.deg2rad(np.arange(360)[valid])+heading
        hits=origin+ranges[valid,None]*np.column_stack([np.cos(aa),np.sin(aa)])
        cells=np.rint((hits-self.origin)/self.res).astype(int)
        for x,y in set(map(tuple,cells)):
            if self.inside((y,x)): self.odds[y,x]=min(4.,self.odds[y,x]+1.5)
        # Known initial/current occupied footprint is clear, except measured obstacles.
        under=np.linalg.norm(self.xy-pose[:2],axis=-1)<.28
        self.odds[under & (self.odds<=0)]=-4.
        occupied=self.odds>.5
        unknown=abs(self.odds)<.25
        self.safe=~(self.dilate(occupied,self.padding)|self.dilate(unknown,self.unknown_padding))
        self.safe[[0,-1],:]=False;self.safe[:,[0,-1]]=False
        self.visits += (np.linalg.norm(self.xy-pose[:2],axis=-1)<.4)*.05

    def line_clear(self,a,b):
        count=max(2,int(np.ceil(np.linalg.norm(np.asarray(b)-a)/(self.res/3)))+1)
        for xy in np.linspace(a,b,count):
            c=self.cell(xy)
            if not self.inside(c) or not self.safe[c]: return False
        return True

    def plan(self,pose,goal):
        self.plans+=1
        start=self.cell(pose[:2]);target=self.cell(goal)
        if not self.inside(start) or not self.safe[start]:
            self.status='NO FOOTPRINT CLEARANCE';return None
        # Dijkstra gives reachable-space distances for both goals and frontiers.
        distances={start:0.};parent={};queue=[(0.,start)]
        moves=[(dy,dx) for dy in (-1,0,1) for dx in (-1,0,1) if dy or dx]
        while queue:
            cost,c=heapq.heappop(queue)
            if cost>distances[c]+1e-9:continue
            if c==target:break
            for dy,dx in moves:
                nb=(c[0]+dy,c[1]+dx)
                if not self.inside(nb) or not self.safe[nb]:continue
                if dy and dx and (not self.safe[c[0]+dy,c[1]] or not self.safe[c[0],c[1]+dx]):continue
                nc=cost+self.res*np.hypot(dx,dy)
                if nc<distances.get(nb,np.inf):
                    distances[nb]=nc;parent[nb]=c;heapq.heappush(queue,(nc,nb))
        if target in distances:
            end=target;self.status='GOAL ROUTE'
        else:
            unknown=abs(self.odds)<.25
            frontier=self.dilate(unknown,self.padding+.24)&self.safe
            candidates=[c for c in distances if frontier[c] and distances[c]>.45]
            if not candidates:
                self.status='NO REACHABLE FRONTIER';return None
            end=min(candidates,key=lambda c:np.linalg.norm(self.xy[c]-goal)+.35*distances[c]+.6*self.visits[c])
            self.status='EXPLORE'
        cells=[end]
        while cells[-1]!=start:cells.append(parent[cells[-1]])
        path=[pose[:2].copy()]+[self.xy[c].copy() for c in reversed(cells[:-1])]
        if end==target and self.line_clear(path[-1],goal):path.append(np.array(goal))
        # Greedy visibility simplification, always on inflated observed-free grid.
        simplified=[path[0]];i=0
        while i<len(path)-1:
            j=len(path)-1
            while j>i+1 and not self.line_clear(path[i],path[j]):j-=1
            if np.linalg.norm(path[j]-simplified[-1])>1e-6:simplified.append(path[j])
            i=j
        if len(simplified)<2:
            self.status='NO PATH PROGRESS';return None
        return np.array(simplified)

    def save(self,path):
        np.savez_compressed(path,log_odds=self.odds,origin=self.origin,
                            resolution=self.res,safe=self.safe,visits=self.visits)
