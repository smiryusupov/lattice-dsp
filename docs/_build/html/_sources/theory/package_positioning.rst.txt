Package positioning
===================

``lattice-dsp`` occupies a narrow intersection inside numerical DSP:

* stable scalar IIR filtering through reflection/PARCOR and lattice-ladder
  coordinates;
* adaptive recursive filtering examples that keep the stability parameterization
  visible;
* AR, Burg, Levinson-Durbin, and spectral diagnostics using the same lattice
  viewpoint;
* finite-Hankel and finite-section Nehari/AAK-style SISO model-reduction
  workflows;
* MIMO block-Hankel reduction, compiled MIMO state-space simulation, and
  matrix-lattice/all-pass scaffolds for multichannel experiments.

The combination is the point.  Many libraries cover general IIR filtering,
general state-space/control workflows, or isolated lattice-filter kernels.  This
package is intentionally focused on the bridge between those areas: stable IIR
lattice coordinates, model-reduction diagnostics, and SISO-to-MIMO tutorial
examples in one C++/Python workflow.

Why the MIMO part is highlighted
--------------------------------

The MIMO material is deliberately separated from scalar IIR examples because the
multichannel case is where the package is most specialized.  The public MIMO
scope includes:

* diagonal-MIMO sanity checks that reduce to independent SISO filters;
* dense coupled state-space examples with Markov-parameter responses;
* finite block-Hankel/ERA-style MIMO reduction;
* a compiled batched state-space runtime for reduced/full MIMO model reuse;
* matrix-lattice all-pass and paraunitary examples;
* bridge diagnostics that compare reduced MIMO Markov data with stable
  matrix-lattice all-pass scaffolds.

This is an uncommon combination for a Python DSP package.  The documentation
therefore uses language such as *focused*, *specialized*, *uncommon*, and *rare
combination* rather than unsupported absolute claims such as *the only* or
*complete*.  The validated scope is finite-dimensional and diagnostic unless a
page explicitly states otherwise.

What is not claimed
-------------------

The positioning above does not mean the package is a complete MIMO solver stack.
The following remain outside the public scope of this release:

* exact infinite-dimensional SISO AAK/Nehari reduction;
* matrix-valued AAK/Nehari optimal reduction;
* constructive dynamic realization of arbitrary MIMO gain responses as matrix
  lattice/all-pass systems;
* production acoustic echo cancellation, room-acoustics, or wireless-precoding
  systems.

A useful way to read the package is therefore:

.. code-block:: text

   SISO lattice IIR primitives
      -> adaptive and spectral lattice diagnostics
      -> finite SISO Hankel/Nehari reduction baselines
      -> MIMO block-Hankel/state-space baselines
      -> matrix-lattice all-pass scaffolds and bridge diagnostics

That chain is the package's niche.
