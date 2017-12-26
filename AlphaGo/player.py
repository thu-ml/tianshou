import argparse
import Pyro4
from game import Game
from engine import GTPEngine

@Pyro4.expose
class Player(object):
    """
    This is the class which defines the object called by Pyro4 (Python remote object).
    It passes the command to our engine, and return the result.
    """
    def __init__(self, **kwargs):
        self.role = kwargs['role']
        self.engine = kwargs['engine']

    def run_cmd(self, command):
        return self.engine.run_cmd(command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--role", type=str, default="unknown")
    parser.add_argument("--debug", type=str, default=False)
    parser.add_argument("--game", type=str, default=False)
    args = parser.parse_args()

    if args.checkpoint_path == 'None':
        args.checkpoint_path = None
    game = Game(name=args.game, role=args.role,
                checkpoint_path=args.checkpoint_path,
                debug=eval(args.debug))
    engine = GTPEngine(game_obj=game, name='tianshou', version=0)

    daemon = Pyro4.Daemon()                # make a Pyro daemon
    ns = Pyro4.locateNS()                  # find the name server
    player = Player(role=args.role, engine=engine)
    print("Init " + args.role + " player finished")
    uri = daemon.register(player)          # register the greeting maker as a Pyro object
    print("Start on name " + args.role)
    ns.register(args.role, uri)            # register the object with a name in the name server
    print("Start requestLoop " + str(uri))
    daemon.requestLoop()                   # start the event loop of the server to wait for calls

