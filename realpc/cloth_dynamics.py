"""Current-frame observation forces on a sewn MuJoCo flex garment.

No animation or recorded cloth trajectories are embedded here. Constraints are
finite-tolerance; strain, contact and grasp residuals must be audited separately.
"""
import numpy as np
import mujoco


def feasible_pin_targets(targets, plane, clearance=.0006):
    """Project each pinch cluster above the table without crushing its layers."""
    from scipy.sparse.csgraph import connected_components
    targets=np.asarray(targets).copy()
    if not len(targets):return targets,0.
    normal=np.r_[-plane[:2],1.];normal/=np.linalg.norm(normal)
    level=plane[2]/np.linalg.norm(np.r_[-plane[:2],1.])
    close=np.linalg.norm(targets[:,None]-targets[None],axis=-1)<.03
    count,labels=connected_components(close,directed=False)
    adjustment=0.
    for group in range(count):
        ids=labels==group
        delta=max(0.,clearance-float(np.min(targets[ids]@normal-level)))
        targets[ids]+=delta*normal;adjustment=max(adjustment,delta)
    return targets,adjustment


class ClothDynamics:
    def __init__(self, rest, faces, X, floor, table_plane=None):
        fmt=lambda x:' '.join(map(str,np.asarray(x).ravel()))
        plane=np.array([0.,0.,floor]) if table_plane is None else np.asarray(table_plane)
        self.table_plane=plane.copy()
        normal=np.r_[-plane[:2],1.]
        if np.min(X@normal-plane[2])<-.00001:
            raise ValueError('Dynamics must start above the actual table plane')
        targets=''.join(f'<body name="target{i}" mocap="true" pos="0 0 1"/>' for i in range(8))
        constraints=''.join(f'<connect name="pin{i}" body1="shirt_0" body2="target{i}" anchor="0 0 0" active="false" solref=".002 1" solimp=".999 .9999 .0001"/>' for i in range(8))
        self.xml=f'''<mujoco>
          <option timestep=".001" integrator="discrete" solver="CG" iterations="100" tolerance="1e-8" jacobian="sparse"><flag ipc="enable"/></option>
          <size memory="2G"/>
          
          <worldbody>
            <geom type="plane" size="2 2 .01" pos="0 0 {plane[2]}" zaxis="{fmt(normal)}" friction=".6 .01 .001" solref=".002 1"/>
            {targets}
            <flexcomp name="shirt" type="direct" dim="2" point="{fmt(rest)}" element="{fmt(faces)}" radius=".0005" mass=".12">
              <edge equality="true" solref=".002 1" solimp=".999 .9999 .0001"/>
              <contact selfcollide="auto" condim="1" solref=".002 1" solimp=".99 .999 .0001"/>
            </flexcomp>
          </worldbody><equality>{constraints}</equality></mujoco>'''
        self.model=m=mujoco.MjModel.from_xml_string(self.xml);self.data=d=mujoco.MjData(m)
        self.ids=m.flex_vertbodyid.copy();self.qadr=m.jnt_qposadr[m.body_jntadr[self.ids]]
        self.dadr=m.jnt_dofadr[m.body_jntadr[self.ids]]
        self.eq=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_EQUALITY,f'pin{i}') for i in range(8)])
        self.qidx=self.qadr[:,None]+np.arange(3);self.didx=self.dadr[:,None]+np.arange(3)
        d.qpos[self.qidx]=X-rest
        mujoco.mj_forward(m,d)
        self.previous_pins={}

    def advance(self,A,rhs,pi,pt,duration=1/30,gain=1.):
        m,d=self.model,self.data
        # FK can compress fingertips through the measured support by millimeters.
        # Keep the material pinch feasible; report its residual to the raw FK.
        pt,self.pin_table_adjustment=feasible_pin_targets(pt,self.table_plane)
        if len(pi)>8:raise ValueError('Use material grasp centers (at most eight)')
        d.eq_active[self.eq]=False
        start=[]
        for i,(vertex,target) in enumerate(zip(pi,pt)):
            eq=self.eq[i];m.eq_obj1id[eq]=self.ids[vertex];m.eq_data[eq,:6]=0
            d.eq_active[eq]=True
            start.append(self.previous_pins.get(int(vertex),d.flexvert_xpos[vertex].copy()))
        start=np.asarray(start).reshape(-1,3)
        steps=max(1,round(duration/m.opt.timestep))
        for s in range(steps):
            X=d.flexvert_xpos.copy()
            # Observation objective becomes a damped force, never a position reset.
            mass=m.body_mass[self.ids]
            damping=2*np.sqrt(mass*gain)
            velocity=d.qvel[self.didx]
            force=gain*(rhs-A@X)/np.maximum(A.diagonal(),1)[:,None]-damping[:,None]*velocity
            # Native IPC has frictionless flex/table contact. Add bounded
            # tangential friction, using only the current support and velocity.
            normal=np.r_[-self.table_plane[:2],1.];normal/=np.linalg.norm(normal)
            level=self.table_plane[2]/np.linalg.norm(np.r_[-self.table_plane[:2],1.])
            on_table=X@normal-level<.0012
            tangent_v=velocity-(velocity@normal)[:,None]*normal
            tangent_f=force-(force@normal)[:,None]*normal
            wanted=-(mass[:,None]*tangent_v/m.opt.timestep+tangent_f)
            limit=.6*(mass*9.81+np.maximum(0.,-force@normal))
            friction=wanted*np.minimum(1.,limit/np.maximum(np.linalg.norm(wanted,axis=1),1e-12))[:,None]
            force[on_table]+=friction[on_table]
            d.qfrc_applied[self.didx]=force
            d.mocap_pos[:len(pi)]=start+(pt-start)*(s+1)/steps
            mujoco.mj_step(m,d)
        mujoco.mj_forward(m,d)
        self.previous_pins={int(i):t.copy() for i,t in zip(pi,pt)}
        self.contacts=int(d.ncon)
        return d.flexvert_xpos.copy()
