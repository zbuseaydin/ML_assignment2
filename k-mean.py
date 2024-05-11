from torchvision import datasets
import PIL 
from PIL import Image
import os
import numpy as np
import argparse
from sklearn.decomposition import PCA


train_dataset = datasets.MNIST(root='./mnist_data', train=True, download=True)
test_dataset = datasets.MNIST(root='./mnist_data', train=False, download=True)
sanitized_data_directory = './mnist_data/sanitized'

args = argparse.ArgumentParser()
args.add_argument("--similarity_distance", "-s", type=str, default='euclidian', help= ['euclidian', 'cosine'])
args.add_argument("--pca", "-p", type=bool, default=False, help="Should PCA be applied to the images?")
args = args.parse_args()


def sanitize_data():
    numbers = [2, 3, 8, 9]
    counters = {2:0, 3:0, 8:0, 9:0}
    try:
        os.mkdir(sanitized_data_directory)
    except:
        return
    for i in range(len(train_dataset)):
        img = train_dataset[i][0]
        label = train_dataset[i][1]
        if label in numbers:
            name = sanitized_data_directory + '/' + str(label) + '_' + str(counters[label]) + '.png'
            img.save(name)
            counters[label] += 1
            

def load_images():
    images = {}
    for filename in os.listdir(sanitized_data_directory):
        img = Image.open(os.path.join(sanitized_data_directory, filename))
        img_array = np.array(img)
        #img_array = img_array/255.0
        images[filename] = img_array.flatten()
        
    return images


def initialize_centroids(images, k =4):
    """
        Randomly select k images from the images dictionary, returns centroids as a list of flattened arrays
        returns: list of flattened arrays
        images: dictionary of image names and their flattened arrays
    """
    centroid_ids = np.random.choice(list(images.keys()), k, replace = False)
    return [images[centroid_id] for centroid_id in centroid_ids]
    


def assign_clusters(images, centroids, similarity_distance = 'euclidian'):
    """
        Assigns each image to the closest centroid
        returns: list of lists, each list contains the image names of the images assigned to the corresponding centroid
        images: dictionary of image names and their flattened arrays
        centroids: list of flattened arrays
    """
    clusters = [[] for i in range(len(centroids))]
    for image_name, image in images.items():
        if similarity_distance == 'euclidian':
            distances = [np.linalg.norm(image - centroid) for centroid in centroids]
            min_centroid = np.argmin(distances)
        elif similarity_distance == 'cosine':
            distances = [np.dot(image, centroid) / (np.linalg.norm(image) * np.linalg.norm(centroid)) for centroid in centroids]
            min_centroid = np.argmax(distances)

        
        clusters[min_centroid].append(image_name)
    

    return clusters
        

def update_centroids(images, centroids):
    """

        images: dictionary of image names and their flattened arrays
        centroids: list of flattened arrays
        clusters: includes image names
    """
    new_centroids = []
    m = [{} for _ in range(len(centroids))] 
    # calculate m by checking the closest centroid for each image
    for image_name, image in images.items():
        distances = [np.linalg.norm(image - centroid) for centroid in centroids]
        closest_centroid = np.argmin(distances)
        m[closest_centroid][image_name] = 1
        for i in range(len(centroids)):
            if i != closest_centroid:
                m[i][image_name] = 0
        
    for i in range(len(centroids)):
        new_centroid = np.zeros(images[list(images.keys())[0]].shape) #initialize new centroid, images[list(images.keys())[0]].shape = 28 * 28 = 784
        for image_name, image in images.items():
            new_centroid += m[i][image_name] * image
        new_centroid = new_centroid / sum(m[i].values())
        new_centroids.append(new_centroid)

    return new_centroids

def is_converged(centroids, new_centroids):
    return np.array_equal(centroids, new_centroids)


def calculate_sse(images, centroids, clusters):
    sse = 0
    for i in range(len(centroids)):
        for image_name in clusters[i]:
            sse += np.linalg.norm(images[image_name] - centroids[i])
    return sse

def calculate_counts_in_clusters(clusters):
    cluster_counters = []
    for cluster in clusters:
        counters = {}
        for image_name in cluster:
            label = int(image_name.split('_')[0])
            if label not in counters:
                counters[label] = 0
            counters[label] += 1
        cluster_counters.append(counters)
    
    return cluster_counters
        
def measure_entropy(clusters):
    entropies = []
    result = 0
    for cluster in clusters:
        total = sum(cluster.values())
        label = max(cluster, key = cluster.get)
        p = cluster[label] / total
        n = 1 - p
        entropy =  -p * np.log(p) - n * np.log(n)
        entropies.append(entropy)
        result += entropy * total
    
    num_images = sum([sum(cluster.values()) for cluster in clusters])
    return result / num_images

        

def calculate_accuracy(clusters):
    correct = 0
    for cluster in clusters:
        label = max(cluster, key = cluster.get)
        correct += cluster[label]
    return correct / sum([sum(cluster.values()) for cluster in clusters])

def apply_pca(images):
    new_images = {}
    pca = PCA(n_components = 128)
    images_list = np.array(list(images.values()))
    images_list = pca.fit_transform(images_list)
    for i, image_name in enumerate(images.keys()):
        new_images[image_name] = images_list[i]
    return new_images


def k_means(images, similarity_distance = 'euclidian'):
    clusters = None
    centroids = initialize_centroids(images)
    while True:
        clusters = assign_clusters(images, centroids,  similarity_distance=similarity_distance)
        new_centroids = update_centroids(images, centroids)
        if is_converged(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids



if __name__ == '__main__':
    sanitize_data()
    similarity_distance = args.similarity_distance
    is_pca = args.pca
    if similarity_distance not in ['euclidian', 'cosine']:
        print('Invalid similarity or distance selection')
        exit(1)

    images = load_images()

    if is_pca:
        images = apply_pca(images)
        
    clusters, centroids = k_means(images, similarity_distance = similarity_distance)
    counts = calculate_counts_in_clusters(clusters)
    acc = calculate_accuracy(counts)
    print("Accuracy: ", round(acc,4))  
    entropy = measure_entropy(counts)
    print("Weighted Entropy", round(entropy,4))
    sse = calculate_sse(images, centroids, clusters)
    print("SSE: ", round(sse,4))



    