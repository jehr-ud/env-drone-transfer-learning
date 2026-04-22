You are an expert in reinforcement learning, control systems, and behavior decomposition.

Your task is to convert a high-level goal into structured "learning definitions" (skills) that can be used by a reinforcement learning agent.

IMPORTANT RULES:

* Do NOT generate vague or abstract skills.
* Each skill MUST be grounded in the available observations and actions.
* Each skill MUST be testable, measurable, and actionable.
* Skills must be decomposed into perception, planning, or control.
* Do NOT invent variables that are not present in the observation space.
* Each skill must include an execution order (integer).
* Output MUST be valid JSON only. No explanations.

---

CRITICAL RULES FOR TRIGGERS (VERY IMPORTANT):

Triggers define WHEN a skill becomes active.

You MUST follow these rules strictly:

1. Triggers must be simple boolean expressions.

2. Use ONLY these operators: <, >, <=, >=, ==, and, or

3. DO NOT use negative thresholds for normalized variables unless explicitly valid.

4. Respect variable ranges:

   * dist ∈ [0, 1]
   * dist_norm ∈ [0, 1]
   * alignment ∈ [-1, 1]
   * rp ∈ [-1, 1]
   * vel_norm ∈ [-1, 1]

5. Use meaningful thresholds:

   * "near obstacle": dist_norm < 0.3
   * "very close obstacle": dist_norm < 0.15
   * "far from goal": dist > 0.5
   * "close to goal": dist < 0.1
   * "misaligned": alignment < 0.8
   * "well aligned": alignment >= 0.9
   * "unstable": abs(rp[0]) > 0.2 or abs(rp[1]) > 0.2

6. Avoid impossible conditions (e.g., dist_norm < 0).

7. Use "always" ONLY for perception or always-on planning.

8. Control skills MUST have meaningful triggers (not always).

9. All triggers MUST be directly evaluable in Python using the provided variables.

   * Triggers will be evaluated using:
     eval(trigger, {}, obs_dict)

   * Therefore:

     * Variables must be used exactly as defined (e.g., dist, alignment, rp)
     * Array indexing is allowed (e.g., rp[0], rp[1])
     * Functions allowed: abs()
     * DO NOT use undefined functions (e.g., min, max, norm)
     * DO NOT use numpy or math functions
     * DO NOT use custom variables

   VALID examples:

   * "dist < 0.1"
   * "alignment < 0.8"
   * "dist_norm < 0.3"
   * "abs(rp[0]) > 0.2 or abs(rp[1]) > 0.2"
   * "alignment < 0.9 and dist > 0.1"

   INVALID examples:

   * "norm(goal_rel) < 1"        # ❌ function not available
   * "distance < 0.1"            # ❌ wrong variable name
   * "dist_norm < -0.2"          # ❌ invalid range
   * "np.linalg.norm(...) < 1"   # ❌ not allowed

---

ENVIRONMENT DESCRIPTION

The agent is a drone navigating in a 3D environment.

GOAL:
Move from point A to point B while avoiding obstacles.

---

OBSERVATION SPACE (normalized values in [-1, 1]):

* goal_rel (3): relative vector to goal (x, y, z)
* vel_norm (3): velocity vector
* rp (2): roll and pitch
* yaw (2): sin(yaw), cos(yaw)
* ang_vel_norm (3): angular velocity
* alignment (1): dot product between velocity and goal direction
* dist (1): normalized distance to goal

OBSTACLES (top 2 closest):
For each obstacle:

* rel_vec (3): relative position
* dist_norm (1): normalized distance
* size_norm (1): normalized size

---

ACTION SPACE:

The agent outputs continuous motion control (vx, vy, vz).

---

REWARD SIGNAL (for reference):

* Positive reward for progress toward goal
* Positive reward for alignment with goal direction
* Positive reward for being closer to goal
* Penalty for not moving
* Penalty for time
* Bonus for reaching goal

---

TASK:

Generate a structured decomposition of skills required to solve the task.

Each skill must follow this format:

{
"order": int,
"name": string,
"type": "perception" | "planning" | "control",
"description": string,
"inputs": [list of observation variables],
"outputs": [list of intermediate or action variables],
"trigger": string,
"objective": string
}

---

OUTPUT FORMAT:

{
"task": "navigate_to_goal_with_obstacle_avoidance",
"skills": [
{ ... }
]
}

---

CONSTRAINTS:

* Use ONLY variables from the observation space
* Skills must be modular and reusable
* Avoid redundancy
* Include both goal-reaching and obstacle-avoidance behaviors

---

NOW GENERATE THE SKILLS.
