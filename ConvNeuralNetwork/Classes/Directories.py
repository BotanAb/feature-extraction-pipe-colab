class Directory:

    def __init__(self, standard):
        self.standard = standard
        self.dir_dict = {}

    def get_standard(self):
        return self.standard

    def get_dir_dict(self, feature):
        return self.dir_dict[feature]

    def add_directory(self, feature, dir):
        self.dir_dict[feature] = dir

    def set_new_directories(self, features):

        if features == "photo to predict":
            dir = input("directory for " + features)+"C:"
            self.add_directory(features, dir)

        else:
            for feature in features:
                dir = input("directory for "+feature)+"C:"
                self.add_directory(feature, dir)

    def set_standard_directories(self, features):

        print(features)
        if features == "photo to predict":
            print("Predict")
            dir = "./"
            self.add_directory(features[0],dir)

        else:
            for feature in features:
                dir = './Videos/' + feature + '/*.mp4'
                self.add_directory(feature, dir)