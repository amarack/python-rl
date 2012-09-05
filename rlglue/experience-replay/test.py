import expreplay
import sarsa_saver

agent = sarsa_saver.sarsa_agent()
expr = expreplay.experience_replay("sarsa_saver.pickle", agent)
