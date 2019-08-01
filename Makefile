all:
	cd alignment/RetinaFace_Mx/rcnn/cython/; python setup.py build_ext --inplace; rm -rf build; cd ../../
	cd alignment/RetinaFace_Mx/rcnn/pycocotools/; python setup.py build_ext --inplace; rm -rf build; cd ../../
clean:
	cd alignment/RetinaFace_Mx/rcnn/cython/; rm *.so *.c *.cpp; cd ../../
	cd alignment/RetinaFace_Mx/rcnn/pycocotools/; rm *.so; cd ../../
