from visualization import visualize


def main():
    print("Select dataset to visualize:")
    print("1: 1D curve in 2D")
    print("2: Circle")
    print("3: Swiss Roll")
    choice = input("Enter choice (1/2/3): ")

    if choice == "1":
        visualize.plot_line_in_2d()
    elif choice == "2":
        visualize.plot_circle()
    elif choice == "3":
        visualize.plot_swiss_roll()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()