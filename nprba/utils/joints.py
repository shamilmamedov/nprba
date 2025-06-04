from enum import Enum, auto


class JointType(Enum):
    FIXED = auto()
    U_ZY = auto()
    P_X = auto()
    P_Y = auto()
    P_Z = auto()
    P_XYZ = auto()
    FREE = auto()


JOINTTYPE_TO_NQ = {
    JointType.FIXED: 0,
    JointType.U_ZY: 2,
    JointType.P_X: 1,
    JointType.P_Y: 1,
    JointType.P_Z: 1,
    JointType.P_XYZ: 3,
    JointType.FREE: 6
}