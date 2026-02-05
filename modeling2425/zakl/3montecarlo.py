"""
Евгений любил придумывать простые игры с подбрасыванием монетки. Недавно он придумал
такую игру для двух человек.
Одна партия игры заключается в том, что надо кидать честную монетку до тех пор, пока не
выпадет комбинация “ОХО”, где “О” — это “Орёл”, а “Х” — это либо “Орёл”, либо “Решка”.
Если за 9 бросков такая комбинация выпала, то Евгений отдаёт второму игроку 1 конфету. Если
же за 9 бросков такая комбинация не выпала, то второй игрок отдаёт Евгению 4 конфеты.
Помогите Евгению оценить ожидаемый выигрыш конфет после 1000 сыгранных партий игры?
"""

from random import choice
def one_game():
    res = 3
    x,y,z = choice(["0","P"]), choice(["0","P"]), choice(["0","P"])
    while x + z != "00":
        x,y,z = y,z, choice(["0","P"])
        res += 1
    if res <= 9:
        win = False
    else:
        win = True
    return win

num_simulations = 50000
max_games = 1000
all_balance = []

for i in range(num_simulations):
    balance = num_rolls = 0
    while num_rolls < max_games:
        win = one_game()
        if win:
            balance = balance + 4
        else:
            balance = balance - 1
        num_rolls += 1
    all_balance.append(balance)

average_end_balance = sum(all_balance)/len(all_balance)
print("Оценка ожидаемого выигрыша после 1000 партий: " + str(average_end_balance))
