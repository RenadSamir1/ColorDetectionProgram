import cv2
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

img_path = r"C:\Users\Renad\OneDrive\Desktop\College\Semester 4\Data Mining\Project\colorpic.jpg"
img = cv2.imread(img_path)

# declaring global variables (are used later on)
clicked = False
r = g = b = x_pos = y_pos = 0

# Reading csv file with pandas and giving names to each column
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv("colors.csv", names=index, header=None)

# Drop the "hex" column
csv.drop("hex", axis=1, inplace=True)

# Clean the dataset
csv = csv.dropna()  # Remove rows with missing values
csv = csv[csv["R"].between(0, 255)]  # Remove rows where R values are out of range
csv = csv[csv["G"].between(0, 255)]  # Remove rows where G values are out of range
csv = csv[csv["B"].between(0, 255)]  # Remove rows where B values are out of range

# Remove duplicates from the DataFrame
csv.drop_duplicates(subset=["color_name", "R", "G", "B"], inplace=True)

# Reset the index
csv.reset_index(drop=True, inplace=True)

# Fit KMeans clustering algorithm to the data
X = csv[["R", "G", "B"]]
kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto").fit(X)
csv["cluster"] = kmeans.predict(X)


# function to get the cluster centroid of the given color
def get_cluster_center(R, G, B):
    return kmeans.predict([[R, G, B]])[0]


plt.boxplot(csv["cluster"])
plt.ylabel("Cluster")
plt.show()


# function to calculate minimum distance from all colors and get the most matching color
def get_color_name(R, G, B):
    cluster = get_cluster_center(R, G, B)
    df_cluster = csv[csv["cluster"] == cluster]
    minimum = 10000
    for i in range(len(df_cluster)):
        d = (
            abs(R - int(df_cluster.iloc[i]["R"]))
            + abs(G - int(df_cluster.iloc[i]["G"]))
            + abs(B - int(df_cluster.iloc[i]["B"]))
        )
        if d <= minimum:
            minimum = d
            cname = df_cluster.iloc[i]["color_name"]
    return cname


# function to get x,y coordinates of mouse double click
def draw_function(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b, g, r, x_pos, y_pos, clicked
        clicked = True
        x_pos = x
        y_pos = y
        b, g, r = img[y, x]
        b = int(b)
        g = int(g)
        r = int(r)


cv2.namedWindow("image")
cv2.setMouseCallback("image", draw_function)

while True:
    cv2.imshow("image", img)
    if clicked:
        # cv2.rectangle(image, start point, endpoint, color, thickness)-1 fills entire rectangle
        cv2.rectangle(img, (20, 20), (750, 60), (b, g, r), -1)

        # Creating text string to display( Color name and RGB values )
        text = (
            get_color_name(r, g, b) + " R=" + str(r) + " G=" + str(g) + " B=" + str(b)
        )

        # cv2.putText(img,text,start,font(0-7),fontScale,color,thickness,lineType )
        cv2.putText(img, text, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # For very light colours we will display text in black colour
        if r + g + b >= 600:
            cv2.putText(img, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        clicked = False

    # Break the loop when user hits 'esc' key
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
