// ---------------------------------------------------------------
        // Auto-fit the poster to the viewport. The .stage wrapper occupies
        // the *scaled* visual area so scrollbars match what the user sees;
        // the .poster inside is the true 48” × 36” landscape print artifact,
        // scaled down via CSS transform.
        // ---------------------------------------------------------------
        (function fit() {
            const root = document.documentElement;
            // Real print pixel dimensions at 96 dpi:
            const PRINT_W = 48 * 96;  // 4608
            const PRINT_H = 36 * 96;  // 3456
            function apply() {
                const margin = 24;
                const availW = Math.max(
                    window.innerWidth || 0,
                    document.documentElement.clientWidth || 0,
                    800
                ) - margin * 2;
                const availH = Math.max(
                    window.innerHeight || 0,
                    document.documentElement.clientHeight || 0,
                    600
                ) - margin * 2;
                let s = Math.min(availW / PRINT_W, availH / PRINT_H);
                // Clamp to a reasonable on-screen range so a small
                // measurement never collapses the poster to a thumbnail.
                s = Math.max(0.20, Math.min(s, 1));
                root.style.setProperty('--scale', s.toFixed(4));
            }
            apply();
            window.addEventListener('resize', apply);
        })();
