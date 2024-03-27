.PHONY: config
config:
	cmake -S . -B ./build -DCMAKE_BUILD_TYPE=RelWithDebInfo

.PHONY: config-debug
config-debug:
	cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Debug

.PHONY: build
build:
	cmake --build build --config RelWithDebInfo -j

.PHONY: build-sequentially
build-sequentially:
	cmake --build build --config RelWithDebInfo

.PHONY: build-debug
build-debug:
	cmake --build build --config Debug -j

.PHONY: clean
clean:
	rm -rf ./build

.PHONY: run
run:
	./build/mlp_learning_an_image data/images/albert.jpg data/config_mem.json 1000 inference.jpg

.PHONY: run-steps
run-steps:
	./build/mlp_learning_an_image data/images/albert.jpg data/config_mem.json $(STEPS) inference.jpg
