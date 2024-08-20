deps:
	# Re-run at every requirements.in change
	pip-compile requirements.in

docker:
	docker build --tag scrock-image .

test1: docker
	# Everything after `docker run ... scrock-image` will be passed to command line of `python3 -m scrock`
	docker rm scrock --force
	docker run --name scrock --volume $(realpath ../data):/data scrock-image refine_clusters /data/sce_sc_10x_5cl_qc.h5ad

test2: docker
	docker rm scrock --force
	docker run --name scrock --volume $(realpath ../data):/data scrock-image find_doublets /data/sce_sc_10x_5cl_qc.h5ad
