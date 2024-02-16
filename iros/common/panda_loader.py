import numpy as np
from os.path import dirname, join, abspath

import hppfcl
import pinocchio as pin

def load_panda():
        """Load the robot from the models folder.

        Returns:
            rmodel, vmodel, cmodel: Robot model, visual model & collision model of the robot.
        """

        ### LOADING THE ROBOT
        pinocchio_model_dir = join(dirname(str(abspath(__file__))), "../../post-treatment/models")
        model_path = join(pinocchio_model_dir, "franka_description/robots")
        mesh_dir = pinocchio_model_dir
        urdf_filename = "franka2.urdf"
        urdf_model_path = join(join(model_path, "panda"), urdf_filename)

        rmodel, cmodel, vmodel = pin.buildModelsFromUrdf(
            urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
        )

        q0 = pin.neutral(rmodel)

        rmodel, [vmodel, cmodel] = pin.buildReducedModel(
            rmodel, [vmodel, cmodel], [1, 9, 10], q0
        )

        ### CREATING THE SPHERE ON THE UNIVERSE
        OBSTACLE_POSE = pin.SE3.Identity()
        OBSTACLE_POSE.translation = np.array([0.0, 0.0, 0.825])
        OBSTACLE_SPHERE = hppfcl.Sphere(0.35/2.0)
        OBSTACLE_SPHERE_GEOM_OBJECT = pin.GeometryObject(
            "obstacle",
            rmodel.getFrameId("universe"),
            rmodel.frames[rmodel.getFrameId("universe")].parent,
            OBSTACLE_SPHERE,
            OBSTACLE_POSE,
        )
        cmodel.addGeometryObject(OBSTACLE_SPHERE_GEOM_OBJECT)

        OBSTACLE_HALFSPACE = hppfcl.Halfspace(np.array([0.0, 0.0, 1.0]), 0.0)
        OBSTACLE_HALFSPACE_GEOM_OBJECT = pin.GeometryObject(
            "table",
            rmodel.getFrameId("universe"),
            rmodel.frames[rmodel.getFrameId("universe")].parent,
            OBSTACLE_HALFSPACE,
            OBSTACLE_POSE,
        )
        cmodel.addGeometryObject(OBSTACLE_HALFSPACE_GEOM_OBJECT)


        for obs in ["obstacle", "table"]:
            cmodel.addCollisionPair(
                pin.CollisionPair(
                    cmodel.getGeometryId("panda2_rightfinger_0"),
                    cmodel.getGeometryId(obs),
                )
            )
            cmodel.addCollisionPair(
                pin.CollisionPair(
                    cmodel.getGeometryId("panda2_leftfinger_0"),
                    cmodel.getGeometryId(obs),
                )
            )
            cmodel.addCollisionPair(
                pin.CollisionPair(
                    cmodel.getGeometryId("panda2_link7_sc_1"),
                    cmodel.getGeometryId(obs),
                )
            )
            cmodel.addCollisionPair(
                pin.CollisionPair(
                    cmodel.getGeometryId("panda2_link7_sc_4"),
                    cmodel.getGeometryId(obs),
                )
            )


        return rmodel, cmodel
    
    
if __name__=="__main__":
    rmodel, cmodel = load_panda()