import matlab.engine

eng = matlab.engine.start_matlab()
print(eng.sum(matlab.double([1, 2, 3, 4])))
