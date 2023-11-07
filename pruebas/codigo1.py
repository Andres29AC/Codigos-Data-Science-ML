#Testing example
#Librerias de test unitario
import unittest

#Probando:
def suma(a,b):
    return a+b

#Clase de testeo
class Test(unittest.TestCase):
    #Metodo de testeo
    def test_suma(self):
        self.assertEqual(suma(2,3),5)
        self.assertEqual(suma(2,-3),-1)
        self.assertEqual(suma(2,0),2)
        self.assertEqual(suma(0,0),0)
        self.assertEqual(suma(-2,-3),-5)

#Ejecucion de testeo
if __name__ == '__main__':
    unittest.main()

#NOTE: C-x g --> git status con magic
#NOTE: C-x b --> cambiar de buffer
#NOTE: C-c t --> project-file
