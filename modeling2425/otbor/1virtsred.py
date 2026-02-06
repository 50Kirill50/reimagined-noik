gr1, gr2, gr3 = 1000.0, 2000.0, 500.0


for _ in range(1, 50):
    rod1 = gr1*0.25
    rod2 = gr2*0.85
    rod3 = gr3*0.15

    pereh2 = gr1*0.85
    pereh3 = gr2*0.80

    gr1 = rod1 + rod2 + rod3
    gr2 = pereh2
    gr3 = pereh3

print(gr1)
