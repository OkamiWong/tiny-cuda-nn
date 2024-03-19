.PHONY: config
config:
	cmake -S . -B ./build -DCMAKE_BUILD_TYPE=RelWithDebInfo

.PHONY: build
build:
	cmake --build build --config RelWithDebInfo -j

.PHONY: build-verbose
build-verbose:
	cmake --build build --config RelWithDebInfo -j --verbose

.PHONY: clean
clean:
	rm -rf ./build

.PHONY: run
run:
	./build/mlp_learning_an_image data/images/albert.jpg data/config_hash.json 1001
