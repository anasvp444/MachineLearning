import random

number = random.randint(1, 40)
guesses = 0
while guesses < 5:
    guess = int(input("\nEnter an integer from 1 to 40: "))
    guesses += 1
    print("this is your %d guess" % guesses)
    if guess < number:
        print("guess is low")
    elif guess > number:
        print("guess is high")
    elif guess == number:
        break
if guess == number:
    guesses = str(guesses)
    print("You guess it in : ", guesses + " guesses")
else:
    print("Best of luck in next time")
