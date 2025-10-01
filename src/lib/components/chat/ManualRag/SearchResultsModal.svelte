<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { v4 as uuidv4 } from 'uuid';
	import { marked } from 'marked';

	export let show = false;
	export let results: any[] = [];

	let selectedIndices = [];
	let expandedIndices = [];
	const dispatch = createEventDispatcher();

	$: if (show) {
		if (results.length === 0) {
			selectedIndices = [];
			expandedIndices = [];
		} else {
			// Sélectionner le top 3
			const top3Indices = results.slice(0, 3).map((_, index) => index);
			
			// Ajouter tous ceux avec score > 0.4 (s'ils ne sont pas déjà dans top3)
			const highScoreIndices = results
				.map((r, i) => ({ score: r.score, index: i }))
				.filter(item => item.score > 0.4 && !top3Indices.includes(item.index))
				.map(item => item.index);
			
			// Combiner les deux ensembles
			selectedIndices = [...top3Indices, ...highScoreIndices];
			expandedIndices = [];
		}
	}

// SearchResultsModal.svelte
function extractSearchText(mainContent: string): string | null {
    if (!mainContent) return null;
    
    // Retirer les marqueurs markdown
    let text = mainContent.replace(/^#+\s*/gm, '');
    
    // Séparer par double saut de ligne (paragraphes)
    const paragraphs = text.split(/\n\n+/).map(p => p.trim()).filter(p => p.length > 0);
    
    if (paragraphs.length === 0) return null;
    
    // Fonction pour vérifier si un mot est "safe"
    const isSafeWord = (word: string): boolean => {
        return /^[a-zA-Z0-9àâäéèêëïîôùûüÿæœçÀÂÄÉÈÊËÏÎÔÙÛÜŸÆŒÇ]/.test(word) &&
               !word.includes("'") &&
               !word.includes("'");
    };
    
    // Chercher une séquence de 6 mots safe dans chaque paragraphe
    for (const paragraph of paragraphs) {
        const cleanParagraph = paragraph.replace(/\n/g, ' ');
        const words = cleanParagraph.split(/\s+/).filter(w => w.length > 0);
        
        // Chercher une fenêtre glissante de 6 mots consécutifs safe
        for (let i = 0; i <= words.length - 6; i++) {
            const window = words.slice(i, i + 6);
            
            if (window.every(isSafeWord)) {
                // Trouvé une séquence safe !
                return window.join(' ');
            }
        }
        
        // Si pas de séquence de 6, essayer avec 4 mots
        for (let i = 0; i <= words.length - 4; i++) {
            const window = words.slice(i, i + 4);
            
            if (window.every(isSafeWord)) {
                return window.join(' ');
            }
        }
    }
    
    // Aucune séquence safe trouvée
    return null;
}

function buildPdfUrl(baseUrl: string, searchText: string | null): string {
    if (!baseUrl) return '#';
    
    // Si on a du texte de recherche et que c'est un PDF
    if (searchText && baseUrl.toLowerCase().endsWith('.pdf')) {
        const encodedSearch = encodeURIComponent(searchText);
        return `${baseUrl}#:~:text=${encodedSearch}`;
    }
    
    return baseUrl;
}

	function getPageFromUrl(url: string): number | null {
		try {
			const match = url.match(/#page=(\d+)/);
			return match ? parseInt(match[1], 10) : null;
		} catch {
			return null;
		}
	}

	function toggleSelection(index: number) {
		const isSelected = selectedIndices.includes(index);
		if (isSelected) {
			selectedIndices = selectedIndices.filter((i) => i !== index);
		} else {
			selectedIndices = [...selectedIndices, index];
		}
	}

	function toggleExpanded(index: number) {
		const isExpanded = expandedIndices.includes(index);
		if (isExpanded) {
			expandedIndices = expandedIndices.filter((i) => i !== index);
		} else {
			expandedIndices = [...expandedIndices, index];
		}
	}

	function getTruncatedContent(content: string, maxLines: number = 5): { truncated: string; needsExpansion: boolean } {
		const lines = content.split('\n');
		if (lines.length <= maxLines) {
			return { truncated: content, needsExpansion: false };
		}
		return { 
			truncated: lines.slice(0, maxLines).join('\n'),
			needsExpansion: true
		};
	}

	function handleConfirm() {
		const selectedResults = results.filter((_, index) => selectedIndices.includes(index));

		if (selectedResults.length === 0) {
			close();
			return;
		}

		const sourceDocuments = selectedResults.map((result) => {
			const searchText = extractSearchText(result.main_content);
			const pdfUrl = buildPdfUrl(result.file_url, searchText);

			return {
				type: 'text',
				id: uuidv4(),
				name: `Source: ${result.title}`,
				content: result.content_with_context,
				status: 'uploaded',
				url: pdfUrl,
				source: {
					url: pdfUrl,
					name: result.title
				}
			};
		});

		dispatch('manualRagConfirm', sourceDocuments);
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
			class="relative w-full max-w-8xl rounded-lg bg-white p-6 shadow-xl dark:bg-gray-800"
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
					{@const searchText = extractSearchText(result.main_content)}
					{@const pdfUrl = buildPdfUrl(result.file_url, searchText)}
					{@const isExpanded = expandedIndices.includes(i)}
					{@const { truncated, needsExpansion } = getTruncatedContent(result.main_content)}
					{@const contentToShow = isExpanded ? result.main_content : truncated}
					
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
							<div class="min-w-0 flex-1">
								<p class="font-semibold text-gray-800 dark:text-gray-200">
									{result.title}
								</p>
								<div class="mt-1 flex items-center space-x-3 text-xs text-gray-500 dark:text-gray-400">
									<a
										href={pdfUrl}
										target="_blank"
										rel="noopener noreferrer"
										class="truncate hover:underline"
										on:click|stopPropagation
									>
										{result.title}
									</a>

									{#if result.score}
										<span class="rounded-full bg-sky-100 px-2.5 py-0.5 font-medium text-sky-800 dark:bg-sky-900 dark:text-sky-300">
											Relevance: {(result.score * 100).toFixed(1)}%
										</span>
									{/if}
								</div>
								<blockquote class="prose prose-sm mt-2 max-w-none border-l-2 border-gray-300 pl-2 text-gray-600 dark:prose-invert dark:border-gray-500 dark:text-gray-400">
									{@html marked(contentToShow)}
								</blockquote>
								{#if needsExpansion}
									<button
										on:click|stopPropagation={() => toggleExpanded(i)}
										class="mt-2 text-sm font-medium text-sky-600 hover:text-sky-700 dark:text-sky-400 dark:hover:text-sky-300"
									>
										{isExpanded ? '▲ Voir moins' : '▼ Voir plus'}
									</button>
								{/if}
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