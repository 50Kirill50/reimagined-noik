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
