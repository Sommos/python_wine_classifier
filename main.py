import wine_classifier

def main():
    white_wine_dataset_path = "datasets/winequality-white.csv"
    white_wine_classifier_path = "classifiers/white_wine_classifier"
    white_wine_fig_path = "figs/white_wine"

    red_wine_dataset_path = "datasets/winequality-red.csv"
    red_wine_classifier_path = "classifiers/red_wine_classifier"
    red_wine_fig_path = "figs/red_wine"

    wine_classifier.wine_classifier(white_wine_dataset_path, white_wine_classifier_path, "white", white_wine_fig_path)
    wine_classifier.wine_classifier(red_wine_dataset_path, red_wine_classifier_path, "red", red_wine_fig_path)

main()