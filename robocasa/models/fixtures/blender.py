from robocasa.models.fixtures import Fixture


class Blender(Fixture):
    """
    Blender fixture class
    """

    def __init__(self, xml, name="blender", *args, **kwargs):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

    @property
    def nat_lang(self):
        return "blender"


class BlenderLid(Fixture):
    def __init__(self, xml, name="blender_lid", has_free_joints=True, *args, **kwargs):
        if has_free_joints:
            joints = [dict(type="free", damping="0.0005")]
        else:
            joints = None

        super().__init__(
            xml=xml,
            name=name,
            duplicate_collision_geoms=False,
            joints=joints,
            *args,
            **kwargs
        )

    @property
    def nat_lang(self):
        return "blender lid"
