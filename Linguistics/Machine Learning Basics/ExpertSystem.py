from random import choice
from experta import Fact, KnowledgeEngine, Rule

dataset = 100
choices = [choice(["green", "blue", "red"]) for i in range(dataset)]
print(len(choices))

class Info(Fact):
    """Info about what we're passing in"""
    pass

class Reaction(KnowledgeEngine):
    @Rule(Info(color="red"))
    def red_color(self):
        print("Neither passed, nor good enough for MIT.")
    @Rule(Info(color="green"))
    def green_color(self):
        print("Passed! Going to MIT, congratulations.")
    @Rule(Info(color="blue"))
    def blue_color(self):
        print("Passed, but not quite good enough for MIT, go to Cornell instead.")

engine = Reaction()
engine.reset()
for i in range(dataset):
    engine.declare(Info(color=choice(choices)))
    engine.run()