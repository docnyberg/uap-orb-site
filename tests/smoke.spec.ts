import { test, expect } from '@playwright/test';

// Set BASE_URL in your env (e.g., http://localhost:8888 for Netlify dev)
// Defaults to http://localhost:8888 if not provided.
const BASE = process.env.BASE_URL || 'http://localhost:8888';

test.describe('Orb Atlas Viewer – smoke', () => {
    test('Reset clears in one click and restores defaults', async ({ page }) => {
        await page.goto(BASE);
        // Wait for boot to finish
        await page.waitForSelector('#grid .card, #note:has-text("Failed")', { timeout: 60000 });
        // Click Reset
        await page.getByRole('button', { name: 'Reset' }).click();

        // Count badge repopulates (> 0)
        const count = await page.locator('#countBadge').innerText();
        expect(Number(count)).toBeGreaterThan(0);

        // Defaults
        await expect(page.locator('#groupSelect')).toHaveValue('video');
        await expect(page.locator('#sortSelect')).toHaveValue('start_ts');
        await expect(page.locator('#sortDirBtn')).toHaveText('↓');
        await expect(page.locator('#thumbMode')).toHaveValue('object');
        await expect(page.locator('#imgOnly')).toBeChecked();
        await expect(page.locator('#hideDup')).not.toBeChecked();
        await expect(page.locator('#collapseDS')).not.toBeChecked();
        await expect(page.locator('#search')).toHaveValue('');
        await expect(page.locator('#hmin')).toHaveValue('3');
        await expect(page.locator('#hmax')).toHaveValue('360');
    });

    test('Similar narrows results', async ({ page }) => {
        await page.goto(BASE);
        try {
            await page.waitForSelector('#grid .card', { timeout: 10000 });
        } catch (error) {
            // Take a screenshot and log what we actually see
            await page.screenshot({ path: 'debug-loading-state.png' });
            const pageContent = await page.content();
            console.log('Page HTML:', pageContent.substring(0, 1000));
            throw error;
        }

        const initialCount = Number(await page.locator('#countBadge').innerText());
        expect(initialCount).toBeGreaterThan(0);

        // Click first card's Similar button
        const firstSim = page.locator('#grid .card .btn-mini[data-action="sim"]').first();
        await firstSim.click();

        // After similar view, count should be <= 31 (1 + top 30)
        await page.waitForFunction(() => Number(document.querySelector('#countBadge')?.textContent || '0') > 0);
        const afterCount = Number(await page.locator('#countBadge').innerText());
        expect(afterCount).toBeLessThanOrEqual(31);
    });

    test('SVG hidden in Scene; visible & actionable in Object', async ({ page }) => {
        await page.goto(BASE);
        await page.waitForSelector('#grid .card');

        // Switch to Scene
        await page.selectOption('#thumbMode', 'scene');
        await page.waitForTimeout(250); // allow refresh
        await expect(page.locator('#grid .card .btn-mini[data-action="svg"]')).toHaveCount(0);
        await expect(page.locator('#collapseDS')).toBeDisabled();

        // Back to Object
        await page.selectOption('#thumbMode', 'object');
        await page.waitForTimeout(250);
        // SVG button should appear
        await expect(page.locator('#grid .card .btn-mini[data-action="svg"]')).toHaveCountGreaterThan(0);

        // Click first SVG to open modal in SVG view
        const svgBtn = page.locator('#grid .card .btn-mini[data-action="svg"]').first();
        await svgBtn.click();
        await expect(page.locator('#modal')).toBeVisible();
        await expect(page.locator('#modalSvgToggle')).toBeChecked();
    });
});