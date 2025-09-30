<script lang="ts">
	import { page } from '$app/state';
import { createEventDispatcher } from 'svelte';
import { v4 as uuidv4 } from 'uuid';

	export let show = false;
	export let results: any[] = [];

	let selectedIndices = [];
	const dispatch = createEventDispatcher();

	$: if (show) {
		selectedIndices = results.map((_, index) => index);
	}

	// This function manually handles adding/removing indices from the array
	function toggleSelection(index: number) {
		const isSelected = selectedIndices.includes(index);

		if (isSelected) {
			// Create a new array without the toggled index
			selectedIndices = selectedIndices.filter((i) => i !== index);
		} else {
			// Create a new array with the added index
			selectedIndices = [...selectedIndices, index];
		}
	}

	function handleConfirm_bu() {
		const selectedResults = results.filter((_, index) => selectedIndices.includes(index));

		if (selectedResults.length === 0) {
			close();
			return;
		}

		let formattedText = '\n\n--- Search Results Context ---\n';
		selectedResults.forEach((result) => {
			formattedText += `\n**Source: ${result.title} (Page ${result.page_number + 1} url: ${result.url})**\n`;
			formattedText += `> ${result.three_page_content.replace(/\n/g, ' ')}\n`;
		});
		formattedText += '--- End of Search Results ---\n\n';

		dispatch('confirm', formattedText);
		close();
	}

    function handleConfirm() {
		const selectedResults = results.filter((_, index) => selectedIndices.includes(index));

		if (selectedResults.length === 0) {
			close();
			return;
		}

		// Create an array of file-like objects for the selected sources
		const sourceDocuments = selectedResults.map(result => ({
			type: 'text',
			id: uuidv4(),
			name: `Source: ${result.title} (Page ${result.page_number + 1})`,
			content: result.three_page_content,
			status: 'uploaded',
            url: `${result.url}#page=${result.page_number + 1}`,
            
            // THIS IS THE CRITICAL ADDITION:
            // We store the original source info here.
            // The backend can use this to construct the citation later.
            source: {
                url: result.url,
                name: result.title
            }
		}));

		dispatch('manualRagConfirm', sourceDocuments); // Dispatch the array of objects
		close();
	}

	function close() {
		show = false;
	}

	const handleKeydown = (e) => {
		if (e.key === 'Escape') {
			close();
		}
	};
</script>

<svelte:window on:keydown={handleKeydown} />

{#if show}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
		on:click={close}
		role="dialog"
		aria-modal="true"
	>
		<div
			class="relative w-full max-w-2xl rounded-lg bg-white p-6 shadow-xl dark:bg-gray-800"
			on:click|stopPropagation
		>
			<div class="mb-4 flex items-center justify-between border-b pb-3 dark:border-gray-700">
				<h3 class="text-xl font-semibold text-gray-900 dark:text-white">Search Results</h3>
				<button
					on:click={close}
					class="rounded-lg p-1.5 text-gray-400 hover:bg-gray-200 hover:text-gray-900 dark:hover:bg-gray-600 dark:hover:text-white"
					aria-label="Close modal"
				>
					<svg class="size-5" fill="currentColor" viewBox="0 0 20 20">
						<path
							fill-rule="evenodd"
							d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
							clip-rule="evenodd"
						/>
					</svg>
				</button>
			</div>

			<div class="max-h-[60vh] space-y-4 overflow-y-auto pr-2">
	{#each results as result, i}
		{@const checkboxId = `result-checkbox-${i}`}
		<label
			for={checkboxId}
			class="block cursor-pointer rounded-lg border p-4 transition-colors hover:bg-gray-50 dark:border-gray-700 dark:hover:bg-gray-700"
		>
			<div class="flex items-start">
				<input
					type="checkbox"
					id={checkboxId}
					class="mr-4 mt-1 size-5 shrink-0 rounded border-gray-300 text-sky-600 focus:ring-sky-500"
					checked={selectedIndices.includes(i)}
					on:change={() => toggleSelection(i)}
				/>
				<div class="min-w-0">
					<p class="font-semibold text-gray-800 dark:text-gray-200">{result.title} (Page: {result.page_number +1})</p>

					<div
						class="mt-1 flex items-center space-x-3 text-xs text-gray-500 dark:text-gray-400"
					>
						<span class="text-gray-300 dark:text-gray-600">|</span>
                        <a
							href={`${result.url}#page=${result.page_number +1}`}
							target="_blank"
							rel="noopener noreferrer"
							class="truncate hover:underline"
							on:click|stopPropagation
						>
							{result.url}
						</a>
					</div>
					<blockquote
						class="mt-2 border-l-2 border-gray-300 pl-2 text-sm text-gray-600 dark:border-gray-500 dark:text-gray-400"
					>
						"{result.small_chunk_content.trim()}"
					</blockquote>
				</div>
			</div>
		</label>
	{:else}
		<p class="text-gray-500">No results found.</p>
	{/each}
</div>

			<div class="mt-6 flex justify-end space-x-3 border-t pt-4 dark:border-gray-700">
				<button
					on:click={close}
					class="rounded-lg bg-gray-200 px-5 py-2.5 text-sm font-medium text-gray-800 hover:bg-gray-300 dark:bg-gray-600 dark:text-white dark:hover:bg-gray-500"
				>
					Cancel
				</button>
				<button
					on:click={handleConfirm}
					class="rounded-lg bg-sky-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-sky-700"
				>
					Add to Message
				</button>
			</div>
		</div>
	</div>
{/if}