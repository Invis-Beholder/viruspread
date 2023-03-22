# viruspread
This is a visual model that simulates the spread of an infectious disease virus among people.
There are 5 states of people: 0 healthy, 1 latent, 2 infected, 3 recovered, 4 dead.
There are four prevention and control strategies: 0, no control; 1, appeal for vaccination; 2, appeal for vaccination and public place control; 3, appeal for vaccination, public place control and large-scale isolation of close contacts of confirmed infected people. 
The model is written in python, mostly based on numpy library operations, and uses matplotlib library to draw.

这是一个模拟传染病病毒在人群中传播的可视化模型，人群有5种状态：0健康，1潜伏，2感染，3康复，4死亡。
分为四种防控策略：0、不管控 1、呼吁接种疫苗 2、呼吁接种疫苗并进行公共场所管制 3、呼吁接种疫苗、进行公共场所管制 且对确诊的感染者的密接者进行大规模隔离。
模型使用python编写，绝大部分基于numpy库操作，并使用matplotlib库绘图
