![alt text](ATOM.png)


# The Atom Project

Atom is a micro modular robot designed as a test platform for studying swarm robotics, including multi-agent planning, control, and coordination. Up to 10 Atom robots can be in play at a given time, acting independently, or as part of a team. Paired with an **overwatch** application, state update information is streamed to each of the robots in a group broadcast, enabling rapid, low cost localization of each bot within it's environment. 


## Communications

Each Atom is connected to a local Wifi network. 

## Localization

Localization of the system is handled by an overwatch system, which identifies Apriltags attached to the compute-chunk of each robot, and to the base station.