# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
"""Run a Twitter simulation against a model served by Unsloth.

Before running:

    1. unsloth run --model unsloth/gemma-3-4b-it-GGUF:Q4_K_M
       (or start Unsloth Studio / Desktop and load a model)
    2. export UNSLOTH_BASE_URL=http://localhost:8000   # port shown by Unsloth
       export UNSLOTH_API_KEY=sk-unsloth-...            # key shown by Unsloth
    3. python examples/twitter_simulation_unsloth.py
"""
import asyncio
import os

import oasis
from oasis import (ActionType, LLMAction, ManualAction,
                   generate_twitter_agent_graph, make_unsloth_model)


async def main():
    # Define the model for the agents: the one currently loaded in Unsloth.
    # Reads UNSLOTH_BASE_URL and UNSLOTH_API_KEY from the environment; the
    # model name is discovered from the server unless UNSLOTH_MODEL is set.
    unsloth_model = make_unsloth_model()

    # Define the available actions for the agents
    available_actions = ActionType.get_default_twitter_actions()

    agent_graph = await generate_twitter_agent_graph(
        profile_path=("data/twitter_dataset/anonymous_topic_200_1h/"
                      "False_Business_0.csv"),
        model=unsloth_model,
        available_actions=available_actions,
    )

    # Define the path to the database
    db_path = "./data/twitter_simulation_unsloth.db"
    os.environ["OASIS_DB_PATH"] = os.path.abspath(db_path)

    # Delete the old database
    if os.path.exists(db_path):
        os.remove(db_path)

    # Make the environment
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path,
    )

    # Run the environment
    await env.reset()

    actions_1 = {}

    actions_1[env.agent_graph.get_agent(0)] = ManualAction(
        action_type=ActionType.CREATE_POST,
        action_args={"content": "Earth is flat."})
    await env.step(actions_1)

    actions_2 = {
        agent: LLMAction()
        # Activate 5 agents with id 1, 3, 5, 7, 9
        for _, agent in env.agent_graph.get_agents([1, 3, 5, 7, 9])
    }

    await env.step(actions_2)

    actions_3 = {}

    actions_3[env.agent_graph.get_agent(1)] = ManualAction(
        action_type=ActionType.CREATE_POST,
        action_args={"content": "Earth is not flat."})
    await env.step(actions_3)

    actions_4 = {
        agent: LLMAction()
        for _, agent in env.agent_graph.get_agents()
    }
    await env.step(actions_4)

    # Close the environment
    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
