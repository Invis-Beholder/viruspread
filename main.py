import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# 5种状态，0健康，1潜伏，2感染，3康复，4死亡
# 4种策略，0123
# todo：隔离区程序 infect_possible里写密接者被隔离 move里写隔离完成后人位置random
# todo：公共场所判定

# People类变量：strategy,round, count, first_infected_count,T_infect,migrant,quarantine
# self参数是一个指向实例本身的引用，用于访问类中的属性和方法。
class People(object):
    def __init__(self, strategy, total_round=300, count=1000, first_infected_count=3, T_infect=20, migrant=10,
                 quarantine=0):
        self.strategy = strategy  # 防控策略
        self.totalround = total_round  # 最大迭代次数
        self.round = 0
        self.count = count  # 总人数
        self.first_infected_count = first_infected_count  # 0代病人数量
        self.T_infect = T_infect  # 感染周期，过了这段时间会死亡或者转为康复
        self._people = np.random.normal(0, 50, (self.count, 2))  # 这里是各个人的坐标位置
        self.segregate = 0
        self.reset()
        self.quarantine_array = []
        self.infected_array = []
        self.healthy_array = []
        self.ill_array = []
        self.recovered_array = []
        self.dead_array = []

        if strategy == 3:
            self._quarantinepeople = np.zeros(())  # 密接者数组
        self.migrant = migrant  # 流动指数
        self.quarantine = quarantine  # 隔离速度
        if strategy > 0:
            self.vaccine_max = min(strategy, 2)  # 接种率上限

    # 在类中定义函数。同__init__()方法一样，实例方法的第一参数必须是self，并且必须包含一个self参数。
    def reset(self):
        self.round = 0
        self._status = np.array([0] * self.count)
        self._timer = np.array([0] * self.count)
        # 初始3个人感染状态
        self.random_people_state(self.first_infected_count, 2)

    def random_people_state(self, num, state=1):
        """ 随机挑选人设置状态
        """
        assert self.count > num
        n = 0
        while n < num:
            i = np.random.randint(0, self.count)
            if self._status[i] == state:
                continue
            else:
                self._status[i] = state
                # 记录状态改变的时间
                self._timer[i] = self.round
                n += 1

    # def set_state(self, i, state):
    #     self._status[i] = state
    #     # 记录状态改变的时间
    #     self._timer[i] = self.round

    def random_movement(self):
        """随机生成移动距离，健康人群乱走，感染人群上移向医院方向，感染距离与migrant参数有关
        """
        w = np.random.normal(0, self.migrant, (self.count, 2))
        for i in range(self.count):
            if self._status[i] != 2 and self._people[i][0] > 250:
                w[i][0] = np.random.uniform(-15, -25)
        # 下面是使感染者向右方医院移动，如果人已经在医院范围内了，x坐标就不动,暂时设定医院区域为x>250的部分
        # 生成和w[self._status==2]一样大小（shape）的数组赋给w[self._status==2]
        w[self._status == 2, 0] = np.random.uniform(20, 30, size=w[self._status == 2].shape[0])
        condition = np.logical_and(self._status == 2, self._people[:, 0] > 250)
        w[condition, 0] = 0
        # 下面是康复者向其他区域定向移动
        dt = self.round - self._timer
        condition = np.logical_and(self._status == 3, dt <= 7)
        w[condition, 0] = np.random.uniform(-20, -30, size=w[condition].shape[0])
        # 隔离区内人不动
        if (self.strategy == 3):
            for i in range(self.count):
                if (self._people[i][0] > 350):
                    w[i] = 0
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

    # 在类外通过对托管属性的直接操作，从而实现类中指定属性的访问、设置、删除。
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

    def move(self, x=.0):
        movement = self.random_movement()
        # 限定特定状态的人员移动
        switch = self.random_switch(x=1.8)
        # movement[(self._status == 0) | switch == 0] = 0
        movement[switch == 0] = 0
        # 死了就别动
        movement[self._status == 4] = 0
        # 公共场所管控，不允许感染者进入，会把里面的感染者赶出去
        if (self.strategy > 1):
            condition = np.logical_and(self._status == 2, self._people[:, 1] > 0)
            if (np.size(movement[condition, 1])):
                movement[condition, 1] = np.random.uniform(-20, -30)
        self._people = self._people + movement

    def change_state(self):
        dt = self.round - self._timer
        # 更新感染者状态，参考现实情况，死亡率约为5%
        condition = np.logical_and((self._status == 2), ((dt == self.T_infect)))
        self._timer[condition] = self.round
        a = np.random.choice([4, 3], size=self._timer[condition].shape, p=[0.05, 0.95])
        self._status[condition] = a
        # 更新潜伏者状态
        condition = np.logical_and((self._status == 1), ((dt == self.T_infect)))
        self._timer[condition] = self.round
        self._status[condition] = 2

    def infect_possible(self, x=0., safe_distance=10.0):
        """潜伏期和感染者都会按概率感染接近的健康人
        x 的取值参考正态分布概率表，x=0 时感染概率是 50%
        """
        # 这里x会收到疫苗接种率vaccine影响
        if (self.strategy > 0):
            x = x + self.vaccine_max * self.round / self.totalround
        temp = np.append(self.infected, self.latent)
        for j in range(len(temp)):
            if (self.strategy == 3 and np.isin(j, self._quarantinepeople)):
                continue  # 隔离区内不感染
            inf = temp[j]
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
                self._timer[i] = self.round

    def iso(self, safe_distance=10.0):
        # 注意_quarantinepeople数组放的是索引位置
        a = np.zeros(())
        for j in range(len(self.infected)):
            inf = self.infected[j]
            if (self.round - self._timer[j] < self.quarantine):
                continue
            if (inf[0] > 350):
                continue
            dm = (self._people - inf) ** 2
            d = dm.sum(axis=1) ** 0.5
            d1 = d.argsort()
            for i in d1:
                if d[i] >= safe_distance:
                    break  # 后面的人超出影响范围，不用管了
                a = np.append(a, i)
        a = np.unique(a)  # 去重
        vector = np.vectorize(np.int_)
        a = vector(a)
        for i in a:
            self._people[i][0] = np.random.normal(400, 10)  # 新的隔离者送入隔离区随机位置
            self._timer[i] = self.round  # 更新时间状态
        self._quarantinepeople = np.append(self._quarantinepeople, a)  # 合并新老隔离者
        self._quarantinepeople = np.unique(self._quarantinepeople)  # 去重
        self._quarantinepeople = vector(self._quarantinepeople)
        a = len(self._quarantinepeople)

        for i in range(a):  # 已有的隔离者检查是否已到隔离时间
            if (self.round - self._timer[self._quarantinepeople[i]] == 24 and (
                    self._status[self._quarantinepeople[i]] != 4)):
                self._people[self._quarantinepeople[i]] = np.random.normal(0, 50, (1, 2))

    def report(self):
        # 散点图部分
        # 一张画布，上画一张散点图（左）两个统计图(右)
        plt.figure(1)
        a1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
        a2 = plt.subplot2grid((2, 3), (0, 2), rowspan=1, colspan=1)
        a3 = plt.subplot2grid((2, 3), (1, 2), rowspan=1, colspan=1)
        a1.cla()
        a1.set_xlim([-410, 440])
        a1.set_ylim([-410, 410])
        p1 = a1.scatter(self.healthy[:, 0], self.healthy[:, 1], s=1)
        p2 = a1.scatter(self.latent[:, 0], self.latent[:, 1], s=1, c='pink')
        p3 = a1.scatter(self.infected[:, 0], self.infected[:, 1], s=1, c='red')
        p4 = a1.scatter(self.recovered[:, 0], self.recovered[:, 1], s=1, c='green')
        p5 = a1.scatter(self.dead[:, 0], self.dead[:, 1], s=1, c='black')
        a1.legend([p1, p2, p3, p4, p5], ['healthy', 'latent', 'infected', 'recovered', 'dead'], loc='upper right',
                  scatterpoints=1)
        t = "Round: %s, Healthy: %s, Latent: %s,Infected: %s, recovered: %s,dead: %s" % \
            (self.round, len(self.healthy), len(self.latent), len(self.infected), len(self.recovered), len(self.dead))
        a1.text(-180, 420, t, ha='left', wrap=True)
        left, bottom, width, height = (250, -410, 100, 820)
        rect = mpatches.Rectangle((left, bottom), width, height, alpha=0.1, facecolor="red")
        a1.add_patch(rect)
        a1.text(265, 0, 'hospital', fontsize=10, color="red", weight="bold")
        left, bottom, width, height = (350, -410, 90, 820)
        rect = mpatches.Rectangle((left, bottom), width, height, alpha=0.1, facecolor="purple")
        a1.add_patch(rect)
        a1.text(355, 10, 'isolation', fontsize=10, color="purple", weight="bold")
        a1.text(370, -10, 'zone', fontsize=10, color="purple", weight="bold")
        left, bottom, width, height = (-410, 0, 660, 440)
        rect = mpatches.Rectangle((left, bottom), width, height, alpha=0.1, facecolor="green")
        a1.add_patch(rect)
        a1.text(-150, 200, 'public area', fontsize=16, color="green", weight="bold")
        left, bottom, width, height = (-410, -410, 660, 410)
        rect = mpatches.Rectangle((left, bottom), width, height, alpha=0.1, facecolor="blue")
        a1.add_patch(rect)
        a1.text(-150, -200, 'open area', fontsize=16, color="blue", weight="bold")
        # 隔离人数折线图部分：感染人数、隔离人数
        a2.cla()
        a2.set_ylim([0, 1200])
        # x,y都得是数组
        x = np.arange(self.round)
        self.quarantine_array.append(np.count_nonzero((self._people[:, 0] > 330)))
        self.infected_array.append(len(self.infected))
        a2.plot(x, self.quarantine_array, 'k.-', x, self.infected_array, 'r.-')
        a2.legend(["quarantinepeople", "infected"], loc='upper right')
        # 各状态人数占比部分
        a3.cla()
        a3.set_ylim([0, 1000])
        self.healthy_array.append(len(self.healthy))
        self.ill_array.append(len(self.latent) + len(self.infected))
        self.recovered_array.append(len(self.recovered))
        self.dead_array.append(len(self.dead))
        y = [self.healthy_array, self.ill_array, self.recovered_array, self.dead_array ]
        pal = ["#9b59b6", "#e74c3c", '#33ff66', "#34495e"]
        a3.stackplot(x, y, labels=['healthy', 'diseased', 'recovered', 'dead'], colors=pal, alpha=0.4)
        plt.legend(loc='upper left')
        # return plt.gcf()

    def update(self):
        self.change_state()
        self.infect_possible(0)
        self.move(1.99)
        if self.strategy == 3:
            self.iso(10)
        self.round += 1

    def record_information(self):
        self.quarantine_array.append(np.count_nonzero((self._people[:, 0] > 330)))
        self.infected_array.append(len(self.infected))
        self.recovered_array.append(len(self.recovered))
        self.dead_array.append(len(self.dead))


def dynamic_function():
    global totalround, p0, p1, p2, p3
    plt.figure(2)
    # 建立画布--4个策略数据dynamic显示
    x = []
    # 清除画布?
    x = range(0, p0.round)
    plt.subplots_adjust(wspace=0.35, hspace=0.2)
    plt.subplot(2, 2, 1)
    # 折现颜色：0 1 2 3
    plt.plot(x, p0.quarantine_array[:p0.round], 'r-', x, p1.quarantine_array[:p1.round], 'y-', x, p2.quarantine_array[:p2.round],
             'b-', x, p3.quarantine_array[:p3.round], 'g-')
    plt.ylabel("quarantine people")

    plt.subplot(2, 2, 2)
    plt.plot(x, p0.infected_array[:p0.round], 'r-', x, p1.infected_array[:p1.round], 'y-', x, p2.infected_array[:p2.round],
             'b-', x, p3.infected_array[:p3.round], 'g-')
    plt.ylabel("infected people")

    plt.subplot(2, 2, 3)
    plt.plot(x, p0.recovered_array[:p0.round], 'r-', x, p1.recovered_array[:p1.round], 'y-', x, p2.recovered_array[:p2.round],
             'b-', x, p3.recovered_array[:p3.round], 'g-')
    plt.ylabel("recovered people")

    plt.subplot(2, 2, 4)
    plt.plot(x, p0.dead_array[:p0.round], 'r-', x, p1.dead_array[:p1.round], 'y-', x, p2.dead_array[:p2.round],
             'b-', x, p3.dead_array[:p3.round], 'g-')
    plt.ylabel("dead people")

    plt.suptitle('Comparision of 4 strategies')
    plt.figure(2).legend(["strategy 0", "strategy 1", "strategy 2", "strategy 3"], loc='upper right', prop={'size':10})


def loop():
    global condition, count, fig, root, dynamic, p, p0, p1, p2, p3
    if not condition:
        root.after(100, loop)
        return
    if p.round < p.totalround:
        p.update()
        p.report()
        p0.update()
        p1.update()
        p2.update()
        p3.update()
        p0.record_information()
        p1.record_information()
        p2.record_information()
        p3.record_information()
        dynamic_function()
    root.after(50, loop)

def start():
    global condition
    condition = True


def pause():
    global condition
    condition = False


def stop():
    global condition, p
    condition = False
    p.reset()


def submit():
    global p, p0, p1, p2, p3
    strategy = strategy_var.get()
    totalround = totalround_var.get()
    count = count_var.get()
    first_infected_count = first_infected_count_var.get()
    T_infect = T_infect_var.get()
    migrant = migrant_var.get()
    quarantine = quarantine_var.get()

    p = People(strategy=strategy, total_round=totalround, count=count, first_infected_count=first_infected_count,
                T_infect=T_infect, migrant=migrant, quarantine=quarantine)
    # 存储运行数据(缺点：先存储数据导致无法直接快速画图，要等较长时间)
    p0 = People(strategy=0, total_round=totalround, count=count, first_infected_count=first_infected_count,
               T_infect=T_infect, migrant=migrant, quarantine=quarantine)
    p1 = People(strategy=1, total_round=totalround, count=count, first_infected_count=first_infected_count,
               T_infect=T_infect, migrant=migrant, quarantine=quarantine)
    p2 = People(strategy=2, total_round=totalround, count=count, first_infected_count=first_infected_count,
               T_infect=T_infect, migrant=migrant, quarantine=quarantine)
    p3 = People(strategy=3, total_round=totalround, count=count, first_infected_count=first_infected_count,
               T_infect=T_infect, migrant=migrant, quarantine=quarantine)


def clear():
    global canvas, strategy_var, totalround_var, count_var, first_infected_count_var, T_infect_var, migrant_var, quarantine_var
    strategy_var.set("")
    totalround_var.set("")
    count_var.set("")
    first_infected_count_var.set("")
    T_infect_var.set("")
    migrant_var.set("")
    quarantine_var.set("")


def layout():
    global root, dynamic
    global strategy_var, totalround_var, count_var, first_infected_count_var, \
        T_infect_var, migrant_var, quarantine_var, dynamic_var
    global right_frame, right_tab1, right_tab2

    root = tk.Tk()
    root.title("Simulator of COVID-19")
    root.geometry("1600x900")
    # root.maxsize(width=1600, height=900)
    # root.config(bg="#FFFFFF")

    left_frame = tk.Frame(root, padx=20, pady=50)
    # left_frame.grid(row=0, column=0)
    left_frame.pack(side='left', fill='y')
    # right_frame = tk.Frame(root)
    s = ttk.Style()
    s.configure('TNotebook.Tab', font=('Consolas', 12, 'normal'))
    right_frame = ttk.Notebook(root)
    # right_frame.grid(row=0, column=1, sticky=tk.N+tk.S+tk.W+tk.E)
    right_frame.pack(side='right', fill='both', expand=True)
    right_tab1 = ttk.Frame(right_frame)
    right_tab2 = ttk.Frame(right_frame)
    right_frame.add(right_tab1, text='tab1')
    right_frame.add(right_tab2, text='tab2')

    para_bar = tk.LabelFrame(left_frame, text="Parameters",
                             font=('Consolas', 24, 'normal'),
                             padx=25, pady=15)
    # para_bar.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N)
    para_bar.pack(side="top", fill="x")
    # input parameters
    # strategy
    strategy_var = tk.IntVar(value=3)
    tk.Label(para_bar, text="strategy", width=20, font=('Consolas', 16, 'normal')).grid(row=0, column=0, padx=10, pady=10)
    tk.Entry(para_bar, textvariable=strategy_var, font=('Consolas', 14, 'normal')).grid(row=0, column=1, padx=10, pady=10)
    # totalround
    totalround_var = tk.IntVar(value=300)
    tk.Label(para_bar, text="total round", font=('Consolas', 16, 'normal')).grid(row=1, column=0, padx=10, pady=10)
    tk.Entry(para_bar, textvariable=totalround_var, font=('Consolas', 14, 'normal')).grid(row=1, column=1, padx=10, pady=10)
    # count
    count_var = tk.IntVar(value=1000)
    tk.Label(para_bar, text="count", font=('Consolas', 16, 'normal')).grid(row=2, column=0, padx=10, pady=10)
    tk.Entry(para_bar, textvariable=count_var, font=('Consolas', 14, 'normal')).grid(row=2, column=1, padx=10, pady=10)
    # first_infected_count
    first_infected_count_var = tk.IntVar(value=3)
    tk.Label(para_bar, text="first_infected_count", font=('Consolas', 16, 'normal')).grid(row=3, column=0, padx=10, pady=10)
    tk.Entry(para_bar, textvariable=first_infected_count_var, font=('Consolas', 14, 'normal')).grid(row=3, column=1, padx=10, pady=10)
    # T_infect
    T_infect_var = tk.IntVar(value=20)
    tk.Label(para_bar, text="T_infect", font=('Consolas', 16, 'normal')).grid(row=4, column=0, padx=10, pady=10)
    tk.Entry(para_bar, textvariable=T_infect_var, font=('Consolas', 14, 'normal')).grid(row=4, column=1, padx=10, pady=10)
    # migrant
    migrant_var = tk.IntVar(value=10)
    tk.Label(para_bar, text="migrant", font=('Consolas', 16, 'normal')).grid(row=5, column=0, padx=10, pady=10)
    tk.Entry(para_bar, textvariable=migrant_var, font=('Consolas', 14, 'normal')).grid(row=5, column=1, padx=10, pady=10)
    # quarantine
    quarantine_var = tk.IntVar(value=0)
    tk.Label(para_bar, text="quarantine", font=('Consolas', 16, 'normal')).grid(row=6, column=0, padx=10, pady=10)
    tk.Entry(para_bar, textvariable=quarantine_var, font=('Consolas', 14, 'normal')).grid(row=6, column=1, padx=10, pady=10)

    tool_bar = tk.LabelFrame(left_frame, text="Options",
                             font=('Consolas', 24, 'normal'),
                             padx=25, pady=15)
    # tool_bar.grid(row=1, column=0, sticky=tk.W+tk.E+tk.S)
    tool_bar.pack(side="bottom", fill="x")
    tool_bar_mid = tk.Frame(tool_bar)
    tool_bar_mid.grid(column=1)
    tool_bar.grid_columnconfigure(0, weight=1)
    tool_bar.grid_columnconfigure(2, weight=1)
    # tool_bar.pack(side='right', fill='y', expand=True)
    # Buttons
    tk.Button(tool_bar_mid, text="Start", command=start, bg="#A1A9D0", fg="black", height=2, width=10, font=('Consolas', 14, 'normal')) \
        .grid(row=0, column=0, padx=20, pady=15)
    tk.Button(tool_bar_mid, text="Pause", command=pause, bg="#F0988C", fg="black", height=2, width=10, font=('Consolas', 14, 'normal')) \
        .grid(row=0, column=1, padx=20, pady=15)
    tk.Button(tool_bar_mid, text="Stop", command=stop, bg="#B883D4", fg="black", height=2, width=10, font=('Consolas', 14, 'normal')) \
        .grid(row=1, column=0, padx=20, pady=15)
    tk.Button(tool_bar_mid, text='Submit', command=submit, bg="#9E9E9E", fg="black", height=2, width=10, font=('Consolas', 14, 'normal')) \
        .grid(row=1, column=1, padx=20, pady=15)
    tk.Button(tool_bar_mid, text='Clear', command=clear, bg="#96CCCB", fg="black", height=2, width=10, font=('Consolas', 14, 'normal')) \
        .grid(row=2, column=0, padx=20, pady=15)


def set_canvas():
    global right_tab1, right_tab2
    canvas1 = FigureCanvasTkAgg(plt.figure(1), master=right_tab1)
    plot_widget1 = canvas1.get_tk_widget()
    plot_widget1.pack(fill='both', expand=True)
    canvas2 = FigureCanvasTkAgg(plt.figure(2), master=right_tab2)
    plot_widget2 = canvas2.get_tk_widget()
    plot_widget2.pack(fill='both', expand=True)


if __name__ == '__main__':
    np.random.seed(0)
    dynamic = 1
    layout()
    set_canvas()
    plt.ion()
    condition = False
    print("PROGRAM START")
    submit()
    root.after(500, loop)
    root.mainloop()
