"""Sandbox runtimes (endpoints): ``guru.sandbox.colima`` drives the Docker
CLI against Colima; ``guru.sandbox.proxy`` runs the provisioning proxy on
an internal network; ``guru.sandbox.provision`` builds the image through
it and turns approved dependency requests into lockfile changes. The spec
and Dockerfile generation live in ``guru.domain.sandbox``, dependency
requests in ``guru.domain.deps``, image records and pending requests in
``guru.repositories.sandbox_images``.
"""
