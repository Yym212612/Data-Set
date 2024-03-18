Normalize the CSV file from the PLAID dataset and obtain a gray vi trajectory. (Here, only part of the PLAID dataset is provided.)

Introduce time characteristics to transform it into a three-dimensional vi trajectory.

Then, convert the three-dimensional gray vi trajectory into color:

Hue (H) represents the "direction" of voltage and current changes between two consecutive points, determined by the rate of change in voltage and current. It is calculated from the angle of change in voltage and current between two consecutive trajectory points (transient feature: the change when a device switches from one state to another).

Saturation (S) represents the power factor, determined by the ratio of active power to apparent power (S ranges from 0.5 to 1) (steady-state feature: distinguishing devices with similar operating modes but different power usage).

Value (V) represents the third harmonic.

Finally, the following trajectory diagram can be obtained

![image](https://github.com/Yym212612/Data-Set/assets/117264647/d1bb2fdb-7bab-45a5-a337-eeb1a1b712b6)
![image](https://github.com/Yym212612/Data-Set/assets/117264647/2790ed0a-9f73-416f-9a5d-c903ce1e0c72)
![image](https://github.com/Yym212612/Data-Set/assets/117264647/c7104d3a-256d-4f54-b655-8db263e8ea7b)
