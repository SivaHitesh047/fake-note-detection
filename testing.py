import unittest
import logging
import warnings
from tensorflow.keras.preprocessing.image import load_img
from Main2 import testing_func

# logging.basicConfig(filename='result_logs.log', format="%(asctime)s %(message)s")
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)


class TestModel(unittest.TestCase):
    def test1(self):
        test_img = load_img("Dataset/Testing/Fake3.png", target_size=(300,300))
        result = testing_func(test_img)
        if result == "Fake":
            print("The note is Fake")
            # logger.info("The note is Fake")
            self.assertEqual(result, "Fake")
        else:
            print("The note is Real")
            # logger.info("The note is Real")
            self.assertEqual(result, "Real")
    
    def test2(self):
        test_img = load_img("Dataset/Testing/Real.jpg", target_size=(300,300))
        result = testing_func(test_img)
        if result == "Fake":
            print("The note is Fake")
            # logger.info("The note is Fake")
            self.assertEqual(result, "Fake")
        else:
            print("The note is Real")
            # logger.info("The note is Real")
            self.assertEqual(result, "Real")

    def test3(self):
        test_img = load_img("Dataset/Testing/Fake2.png", target_size=(300,300))
        result = testing_func(test_img)
        if result == "Fake":
            print("The note is Fake")
            # logger.info("The note is Fake")
            self.assertEqual(result, "Fake")
        else:
            print("The note is Real")
            # logger.info("The note is Real")
            self.assertEqual(result, "Real")

    def test4(self):
        test_img = load_img("Dataset/Testing/Real2.jpg", target_size=(300,300))
        result = testing_func(test_img)
        if result == "Fake":
            print("The note is Fake")
            # logger.info("The note is Fake")
            self.assertEqual(result, "Fake")
        else:
            print("The note is Real")
            # logger.info("The note is Real")
            self.assertEqual(result, "Real")
    
    def test5(self):
        test_img = load_img("Dataset/Testing/Fake.jpg", target_size=(300,300))
        result = testing_func(test_img)
        if result == "Fake":
            print("The note is Fake")
            # logger.info("The note is Fake")
            self.assertEqual(result, "Fake")
        else:
            print("The note is Real")
            # logger.info("The note is Real")
            self.assertEqual(result, "Real")

if __name__ == '__main__':
    unittest.main()
