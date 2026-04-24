You are an expert in control systems and reinforcement learning.

Your task is to generate a base policy for an agent in a continuous control environment.

The policy must be:

- Deterministic
- Interpretable
- Based only on the observation vector
- Expressed as a structured JSON (NOT code)
- Generalizable to different environments

The output must follow this schema:

{
  "name": "policy_name",
  "description": "short description",
  "inputs": [
    {"name": "...", "indices": [start, end]}
  ],
  "intermediate": [
    {
      "name": "...",
      "type": "operation",
      "op": "sub | add | norm | clip | scale | dot",
      "inputs": ["var1", "var2"],
      "params": {}
    }
  ],
  "outputs": [
    {
      "name": "action",
      "components": [
        {"source": "variable_name", "index": i}
      ]
    }
  ]
}

Constraints:
- Use vector operations when possible
- Avoid loops
- Keep it efficient
- Do not generate Python code
- Only JSON

Environment description:
- First 3 values: relative goal position
- Next 3 values: velocity
- Action space: 4D (vx, vy, vz, speed)

Goal:
Move the agent toward the goal while stabilizing motion.