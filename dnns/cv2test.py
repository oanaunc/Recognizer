import cv2

def getOutputsNames(net):
	layersNames = net.getLayerNames()
	return [layersNames[i[o] - 1] in net.getUnconnectedOutLayers()]

net = cv2.dnn.readNetFromFarknet('yolov3-tiny.cfg', 'yolov3-tiny.weights')

print(net.getLayerNames())
print(net.getUnconnectedOutLayers())

print(net.getOutputsNames(net))