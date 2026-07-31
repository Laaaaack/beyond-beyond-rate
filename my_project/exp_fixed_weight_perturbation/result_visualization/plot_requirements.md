# Plot Requirements

These conventions apply to all result-visualization notebooks/scripts in this
directory.

1. **Panel labels.** Always add alphabetic ordering labels (`(a)`, `(b)`,
   `(c)`, ...) at the top-left corner of each subplot, e.g.:

   ```python
   ax.text(-0.06, 1.04, "(a)", transform=ax.transAxes,
           fontsize=14, fontweight="bold")
   ```

2. **Grid.** Always turn the grid on for every subplot, e.g.:

   ```python
   ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
   ```

3. **Legend ordering for Whole/Part/Norm.** Whenever a legend contains the
   dataset variants "Whole", "Part", and "Norm", keep them together in a
   single column, in that order (Whole, Part, Norm). If the legend also has
   line-style entries (e.g. delay vs. no-delay, inverse vs. original), put
   those in a separate column so the three variants stay grouped in their
   own column, e.g.:

   ```python
   blank = mlines.Line2D([], [], linestyle="none", marker="none", label="")
   handles = [
       mlines.Line2D([], [], color="black", marker="o", linestyle="-",
                     markersize=5, linewidth=1.8, label="<style A>"),
       mlines.Line2D([], [], color="black", marker="x", linestyle="--",
                     markersize=6, linewidth=1.8, label="<style B>"),
       blank,
   ] + [
       mlines.Line2D([], [], color=COLORS[variant], linewidth=2.5, label=variant)
       for variant in ("Whole", "Part", "Norm")
   ]
   ax.legend(handles=handles, loc="upper right", framealpha=0.9, ncol=2)
   ```

   The blank third entry is needed because matplotlib fills legend columns
   top-to-bottom: with only 5 real entries it splits 3+2 and pushes "Whole"
   into the style column; the blank pads it to 6 so it splits cleanly 3+3.
