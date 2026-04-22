class TriggerValidator:
    def __init__(self, obs_dict):
        self.obs_dict = obs_dict

    def validate_or_fallback(self, trigger):
        if trigger == "always":
            return trigger

        try:
            result = eval(trigger, {}, self.obs_dict)
            if isinstance(result, bool):
                return trigger
        except Exception as e:
            print(f"[Trigger INVALID] {trigger} -> {e}")

        return "always"