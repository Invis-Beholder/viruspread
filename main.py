import numpy
import numpy as np
import matplotlib.pyplot as plt
#5种状态，0健康，1潜伏，2感染，3康复，4死亡
#4种策略，0123
#todo：隔离区程序 infect_possible里写密接者被隔离 move里写隔离完成后人位置random
# todo：公共场所判定

class People(object):
    def __init__(self,strategy,round=300, count=1000, first_infected_count=3,T_infect=20,migrant=10,quarantine=0):
        self.strategy=strategy#防控策略
        self.totalround=round#最大迭代次数
        self.count = count#总人数
        self.first_infected_count = first_infected_count#0代病人数量
        self.T_infect=T_infect#感染周期，过了这段时间会死亡或者转为康复
        self._people = np.random.normal(0, 50, (self.count, 2))#这里是各个人的坐标位置
        self.reset()
        if(strategy==3):
            self._quarantinepeople=np.zeros(())#密接者数组
        self.migrant=migrant#流动指数
        self.quarantine = quarantine#隔离速度
        if(strategy>0):
            self.vaccine_max=min(strategy,2)#接种率上限
        for i in range(round):#运行部分
            self.update()
            self.report()
            plt.pause(.1)

    def reset(self):
        self._round = 0
        self._status = np.array([0] * self.count)
        self._timer = np.array([0] * self.count)
        self.random_people_state(self.first_infected_count, 2)

    def random_people_state(self, num, state=1):
        """随机挑选人设置状态
        """
        assert self.count > num
        n = 0
        while n < num:
            i = np.random.randint(0, self.count)
            if self._status[i] == state:
                continue
            else:
                self.set_state(i, state)
                n += 1

    def set_state(self, i, state):
        self._status[i] = state
        # 记录状态改变的时间
        self._timer[i] = self._round

    def random_movement(self ):
        """随机生成移动距离，健康人群乱走，感染人群上移向医院方向，感染距离与migrant参数有关
        """
        w=np.random.normal(0, self.migrant, (self.count, 2))
        #下面是使感染者向右方医院移动，如果人已经在医院范围内了，x坐标就不动,暂时设定医院区域为x>250的部分
        # 生成和w[self._status==2]一样大小（shape）的数组赋给w[self._status==2]
        w[self._status == 2, 0] = np.random.uniform(15, 25,size=w[self._status == 2].shape[0])
        condition = np.logical_and(self._status == 2, self._people[:, 0] > 250)
        w[condition, 0] = 0
        #下面是康复者向其他区域定向移动
        dt = self._round - self._timer
        condition=np.logical_and(self._status==3,dt<=7)
        w[condition, 0] =np.random.uniform(-15, -25, size=w[condition].shape[0])
        #隔离区内人不动
        if(self.strategy==3):
            for i in range(self.count):
                if (self._people[i][0]<-200 and self._people[i][1]<-200):
                    w[i]=0
        return w

    def random_switch(self, x=0.):
        """随机生成开关，0 - 关，1 - 开

        x 大致取值范围 -1.99 - 1.99；
        对应正态分布的概率， 取值 0 的时候对应概率是 50%
        :param x: 控制开关比例
        :return:
        """
        normal = np.random.normal(0, 1, self.count)
        switch = np.where(normal < x, 1, 0)
        return switch

    @property
    def healthy(self):
        return self._people[self._status == 0]

    @property
    def latent(self):
        return self._people[self._status == 1]

    @property
    def infected(self):
        return self._people[self._status == 2]

    @property
    def recovered(self):
        return self._people[self._status == 3]

    @property
    def dead(self):
        return self._people[self._status == 4]

    def move(self,  x=.0):
        movement = self.random_movement()
        # 限定特定状态的人员移动
        switch = self.random_switch(x=1.8)
        # movement[(self._status == 0) | switch == 0] = 0
        movement[switch == 0] = 0
        #死了就别动
        movement[self._status==4]=0

        if(self.strategy>1):
            condition = np.logical_and(self._status == 2, self._people[:, 1] >0)
            if(np.size(movement[condition,1])):
                movement[condition,1]=np.random.uniform(-10, -20)
        self._people = self._people + movement

    def change_state(self):
        dt = self._round - self._timer
        # 更新感染者状态，参考现实情况，死亡率约为5%
        condition=np.logical_and((self._status == 2),((dt == self.T_infect) ))
        self._timer[condition] = self._round
        a = np.random.choice([4, 3], size=self._timer[condition].shape, p=[0.05, 0.95])
        self._status[condition]=a
        # 更新潜伏者状态
        condition = np.logical_and((self._status == 1), ((dt == self.T_infect)))
        self._timer[condition] = self._round
        self._status[condition] =2

    def infect_possible(self, x=0., safe_distance=10.0):
        """潜伏期和感染者都会按概率感染接近的健康人
        x 的取值参考正态分布概率表，x=0 时感染概率是 50%
        """
        #这里x会收到疫苗接种率vaccine影响
        if(self.strategy>0):
                x=x+self.vaccine_max*self._round/self.totalround
        temp=np.append(self.infected,self.latent)
        for j in range(len(temp)):
            if (self.strategy == 3 and np.isin(j, self._quarantinepeople)):
                continue  # 隔离区内不感染
            inf=temp[j]
            dm = (self._people - inf) ** 2
            d = dm.sum(axis=1) ** 0.5
            sorted_index = d.argsort()
            for i in sorted_index:
                if d[i] >= safe_distance:
                    break  # 后面的人超出影响范围，不用管了
                if self._status[i] > 0:
                    continue
                if np.random.normal() > x:
                    continue
                self._status[i] = 1
                # 记录状态改变的时间
                self._timer[i] = self._round

    def iso(self,safe_distance=10.0):
        #注意_quarantinepeople数组放的是索引位置
            a=np.zeros(())
            for j in range(len(self.infected)):
                inf = self.infected[j]
                if(self._round-self._timer[j]<self.quarantine):
                    continue
                if(inf[0]<-230 and inf[1]<-230):
                    continue
                dm = (self._people - inf) ** 2
                d = dm.sum(axis=1) ** 0.5
                d1= d.argsort()
                for i in d1:
                    if d[i] >= safe_distance:
                        break  # 后面的人超出影响范围，不用管了
                    a=np.append(a,i)
            a=np.unique(a)#去重
            vector = np.vectorize(np.int_)
            a = vector(a)
            for i in a:
                self._people[i]=np.random.normal(-250, 10, (1, 2))#新的隔离者送入隔离区随机位置
                self._timer[i]=self._round#更新时间状态
            self._quarantinepeople = np.append(self._quarantinepeople, a)  # 合并新老隔离者
            self._quarantinepeople = np.unique(self._quarantinepeople)  # 去重
            self._quarantinepeople=vector(self._quarantinepeople)
            a=len(self._quarantinepeople)
            for i in range(a):#已有的隔离者检查是否已到隔离时间
                if(self._round - self._timer[i]==24 and self._status[i]!=4):
                    self._people[i]=np.random.normal(0, 50, (1, 2))
                    # np.delete(self._quarantinepeople,i)
                    # a=len(self._quarantinepeople)




    def report(self):
        plt.cla()
        # plt.grid(False)
        p1 = plt.scatter(self.healthy[:, 0], self.healthy[:, 1], s=1)
        p2 = plt.scatter(self. latent[:, 0], self.latent[:, 1], s=1, c='pink')
        p3 = plt.scatter(self.infected[:, 0], self.infected[:, 1], s=1, c='red')
        p4 = plt.scatter(self.recovered[:, 0], self.recovered[:, 1], s=1, c='green')
        p5 = plt.scatter(self.dead[:, 0], self.dead[:, 1], s=1, c='black')
        plt.legend([p1, p2, p3,p4,p5], ['healthy', 'latent','infected', 'recovered','dead'], loc='upper right', scatterpoints=1)
        t = "Round: %s, Healthy: %s, Latent: %s,Infected: %s, recovered: %s,dead: %s" % \
            (self._round, len(self.healthy), len(self.infected), len(self.recovered),len(self.latent),len(self.dead))
        plt.text(-200, 400, t, ha='left', wrap=True)

    def update(self):
        """每一次迭代更新"""
        self.change_state()
        self.infect_possible(0)
        self.move( 1.99)
        if(self.strategy==3):
            self.iso(10)
        self._round += 1
        self.report()


if __name__ == '__main__':
    np.random.seed(0)
    plt.figure(figsize=(15, 15), dpi=100)
    plt.ion()
    p = People(3,round=300, count=1000, first_infected_count=3,T_infect=20,migrant=10,quarantine=0)
    plt.pause(3)