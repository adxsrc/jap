|jap|_: map engine for |JAX|_
=============================

|jap|_ is a map engine for Google's |JAX|_.
It maps python functions to
`JAX arrays <https://jax.readthedocs.io/en/latest/jax_array_migration.html>`_ or
`pytree <https://jax.readthedocs.io/en/latest/pytrees.html>`_,
effectively turning python functions into accelerator kernels.
In addition to unifying |JAX|_'s ``vmap()``, ``pmap()``, and
``xmap()`` interfaces, it also makes implementations of, e.g.,
integrators of ordinary differential equations with adaptive time
steps, more straightforward.

.. |jap| replace:: ``jap``
.. |JAX| replace:: ``JAX``

.. _jap: https://github.com/adxsrc/jap
.. _JAX: https://github.com/google/jax
