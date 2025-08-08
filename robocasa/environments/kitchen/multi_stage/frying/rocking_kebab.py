from robocasa.environments.kitchen.kitchen import *


class RockingKebab(Kitchen):
    """
    Fry kebab with Pan Rock: composite task for Frying Foods activity.

    Simulates the process of frying kebab while rocking the pan back and forth.

    Steps:
        1) Pick up a kebab from the plate.
        2) Place the kebab in an empty pan on the stove.
        3) Rock the pan back and forth at least 5 times.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove)
        )

        self.init_robot_base_ref = self.stove

        if "refs" in self._ep_meta:
            self.knob = self._ep_meta["refs"]["knob"]
        else:
            valid_knobs = []

            for knob, joint in self.stove.knob_joints.items():
                if joint is not None and not knob.startswith("rear"):
                    valid_knobs.append(knob)

            self.knob = self.rng.choice(list(valid_knobs))

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick up the kebab from the plate, place it in on the pan on the stove, "
            "then rock the pan back and forth at least 5 times. Make sure to put the pan back on the stove when done."
        )
        ep_meta["knob"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.stove.set_knob_state(env=self, rng=self.rng, knob=self.knob, mode="on")
        self.pan_rocking_timer = 0
        self.pan_rock_count = 0
        self.rocking_done = False
        self.pan_last_sign = None

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="kebab",
                obj_groups="kebab_skewer",
                object_scale=1.15,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.50, 0.30),
                    pos=("ref", -1.0),
                    try_to_place_in="plate",
                ),
            )
        )

        cfgs.append(
            dict(
                name="pan",
                obj_groups="pan",
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    sample_region_kwargs=dict(
                        locs=[self.knob],
                    ),
                    size=(0.05, 0.05),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Success condition:
        1. The kebab must be placed inside the pan.
        2. The correct stove burner must stay on.
        3. The pan must be rocked back and forth at least 5 cycles.
        4. Once it has been rocked back and forth, put the pan back on the stove.
        """
        kebab_in_pan = OU.check_obj_in_receptacle(self, "kebab", "pan")

        pan_on_stove = OU.check_obj_fixture_contact(self, "pan", self.stove)

        if not kebab_in_pan:
            return False

        burner_on = self.stove.is_burner_on(env=self, burner_loc=self.knob)

        pan_joint_id = self.sim.model.get_joint_qpos_addr("pan_joint0")
        pan_velocity = self.sim.data.qvel[pan_joint_id]
        velocity_threshold = 0.07

        if abs(pan_velocity) > velocity_threshold:
            current_sign = np.sign(pan_velocity)
            if self.pan_last_sign is None:
                self.pan_last_sign = current_sign
            elif current_sign != self.pan_last_sign:
                self.pan_rock_count += 1
                self.pan_last_sign = current_sign

        if self.pan_rock_count >= 5:
            self.rocking_done = True

        return (
            kebab_in_pan
            and pan_on_stove
            and burner_on
            and getattr(self, "rocking_done", False)
        )
