<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { v4 as uuidv4 } from 'uuid';
	import { marked } from 'marked';

	export let show = false;
	export let results: any[] = [];
	export let viewOnly = false; // ✅ Nouvelle prop pour le mode lecture seule

	let selectedIndices = [];
	let expandedIndices = [];
	const dispatch = createEventDispatcher();

	$: if (show) {
		if (results.length === 0) {
			selectedIndices = [];
			expandedIndices = [];
		} else {
			// ✅ Tout désélectionné par défaut
			selectedIndices = [];
			expandedIndices = [];
		}
	}

	function buildPreciseUrl(result: any): string {
		const fileUrl = result.file_url;
		const sourceUrl = result.source_url;
		
		// ✅ Détecter le type à partir de file_type OU de l'extension du file_url
		const isPdf = result.file_type === 'pdf' || fileUrl?.toLowerCase().endsWith('.pdf');
		const isHtml = ['html', 'htm'].includes(result.file_type) || 
		               fileUrl?.toLowerCase().endsWith('.html') || 
		               fileUrl?.toLowerCase().endsWith('.htm');
		
		// ========================================
		// LOGIQUE POUR LES PDFs
		// ========================================
		if (isPdf) {
			if (!fileUrl) return '#';
			
			// ✨ STRATÉGIE 1 : Ancre de node (précision maximale)
			if (result.node_anchor_id) {
				return `${fileUrl}#nameddest=${result.node_anchor_id}`;
			}
			
			// ✨ STRATÉGIE 2 : Fallback sur page_number
			if (result.page_number) {
				return `${fileUrl}#page=${result.page_number}`;
			}
			
			// ✨ STRATÉGIE 3 : URL simple (fichier depuis serveur)
			return fileUrl;
		}
		
		// ========================================
		// LOGIQUE POUR LES HTMLs
		// ========================================
		if (isHtml) {
			// ✅ Pour HTML, on utilise source_url (site WordPress avec CSS/JS)
			if (!sourceUrl) return fileUrl || '#';
			
			// ✨ STRATÉGIE 1 : Text fragment avec start,end (vient directement du backend)
			if (result.search_text_start && result.search_text_end) {
				const start = encodeURIComponent(result.search_text_start);
				const end = encodeURIComponent(result.search_text_end);
				
				// ✅ Si start et end sont identiques, utiliser la syntaxe simple
				if (result.search_text_start === result.search_text_end) {
					return `${sourceUrl}#:~:text=${start}`;
				}
				
				// Sinon, utiliser la syntaxe start,end
				return `${sourceUrl}#:~:text=${start},${end}`;
			}
			
			// ✨ STRATÉGIE 2 : Fallback sur node_anchor_id (ancre HTML classique)
			if (result.node_anchor_id) {
				return `${sourceUrl}#${result.node_anchor_id}`;
			}
			
			// ✨ STRATÉGIE 3 : URL simple (site WordPress original)
			return sourceUrl;
		}
		
		// ========================================
		// AUTRES TYPES DE FICHIERS
		// ========================================
		return fileUrl || '#';
	}

	function toggleSelection(index: number) {
		if (viewOnly) return; // ✅ Pas de sélection en mode viewOnly
		
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
			// Scroller vers le haut de l'élément
			setTimeout(() => {
				const element = document.getElementById(`result-item-${index}`);
				if (element) {
					element.scrollIntoView({ behavior: 'smooth', block: 'start' });
				}
			}, 0);
		} else {
			expandedIndices = [...expandedIndices, index];
		}
	}

	// ✅ Fonction pour tronquer le precise_content uniquement
	function getTruncatedPreciseContent(preciseContent: string, maxLines: number = 5): { 
		truncated: string; 
		needsExpansion: boolean 
	} {
		const lines = preciseContent.split('\n');
		if (lines.length <= maxLines) {
			return { truncated: preciseContent, needsExpansion: false };
		}
		return { 
			truncated: lines.slice(0, maxLines).join('\n'),
			needsExpansion: true
		};
	}

	function handleConfirm() {
		if (viewOnly) return; // ✅ Pas de confirmation en mode viewOnly
		
		const selectedResults = results.filter((_, index) => selectedIndices.includes(index));

		if (selectedResults.length === 0) {
			close();
			return;
		}

		const sourceDocuments = selectedResults.map((result) => {
			// ✅ Utiliser buildPreciseUrl pour construire l'URL avec l'ancre appropriée
			const preciseUrl = buildPreciseUrl(result);

			return {
				type: 'text',
				id: uuidv4(),
				name: `Source: ${result.title}`,
				// ✅ Utiliser context_content pour le LLM (plus de contexte)
				content: result.context_content,
				status: 'uploaded',
				url: preciseUrl,
				isRagSource: true,
				source: {
					url: preciseUrl,
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

	function toggleSelectAll() {
		if (viewOnly) return; // ✅ Pas de sélection en mode viewOnly
		
		if (selectedIndices.length === results.length) {
			// Si tout est sélectionné, tout désélectionner
			selectedIndices = [];
		} else {
			// Sinon, tout sélectionner
			selectedIndices = results.map((_, index) => index);
		}
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
		class="fixed inset-0 z-[100] flex items-center justify-center bg-black/50"
		on:click={close}
		role="dialog"
		aria-modal="true"
	>
		<div
			class="relative w-full max-w-8xl mx-4 rounded-lg bg-white p-6 shadow-xl dark:bg-gray-800"
			on:click|stopPropagation
		>
			<div class="mb-4 flex items-center justify-between border-b pb-3 dark:border-gray-700">
				<div class="flex items-center gap-4">
					<h3 class="text-xl font-semibold text-gray-900 dark:text-white">Search Results</h3>
					
					{#if !viewOnly}
						<button
							on:click={toggleSelectAll}
							class="flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700"
						>
							<input
								type="checkbox"
								class="size-4 rounded border-gray-300 text-sky-600 focus:ring-sky-500 pointer-events-none"
								checked={selectedIndices.length === results.length && results.length > 0}
								indeterminate={selectedIndices.length > 0 && selectedIndices.length < results.length}
							/>
							<span>
								{#if selectedIndices.length === results.length && results.length > 0}
									Unselect All
								{:else}
									Select All
								{/if}
							</span>
						</button>
					{/if}
				</div>
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
					{@const isExpanded = expandedIndices.includes(i)}
					
					<!-- ✅ Utilise TOUJOURS precise_content (tronqué ou complet) -->
					{@const { truncated: truncatedPrecise, needsExpansion } = getTruncatedPreciseContent(result.precise_content)}
					{@const contentToShow = isExpanded ? result.precise_content : truncatedPrecise}
					
					<!-- ✅ URL construite avec buildPreciseUrl (PDF ou HTML) -->
					{@const preciseUrl = buildPreciseUrl(result)}
					
					<div
						id="result-item-{i}"
						class="block rounded-lg border p-4 transition-colors hover:bg-gray-50 dark:border-gray-700 dark:hover:bg-gray-700"
					>
						<div class="flex items-start gap-4">
							<!-- ✅ Checkbox visible uniquement si NOT viewOnly -->
							{#if !viewOnly}
								<div
									class="flex shrink-0 items-center justify-center w-12 cursor-pointer self-stretch"
									on:click={() => toggleSelection(i)}
									role="button"
									tabindex="0"
									on:keydown={(e) => e.key === 'Enter' && toggleSelection(i)}
								>
									<input
										type="checkbox"
										id={checkboxId}
										class="size-5 rounded border-gray-300 text-sky-600 focus:ring-sky-500 pointer-events-none"
										checked={selectedIndices.includes(i)}
										on:change={() => toggleSelection(i)}
									/>
								</div>
							{/if}
							
							<!-- ✅ Zone contenu cliquable pour expand/collapse -->
							<div 
								class="min-w-0 flex-1 cursor-pointer"
								on:click={() => toggleExpanded(i)}
								role="button"
								tabindex="0"
								on:keydown={(e) => e.key === 'Enter' && toggleExpanded(i)}
							>
								<div class="flex items-center space-x-3 text-xs text-gray-500 dark:text-gray-400">
									<a
										href={preciseUrl}
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
								
								<!-- ✅ Affichage UNIQUEMENT de precise_content (tronqué ou complet) -->
								<blockquote class="prose prose-sm mt-2 max-w-none border-l-2 border-gray-300 pl-2 text-gray-600 dark:prose-invert dark:border-gray-500 dark:text-gray-400">
									{@html marked(contentToShow)}
								</blockquote>
								
								<!-- ✅ Indicateur visuel en bas -->
								{#if needsExpansion}
									<div class="mt-2 text-sm font-medium text-sky-600 dark:text-sky-400">
										{#if isExpanded}
											▲ Click to show less
										{:else}
											▼ Click to show more 
										{/if}
									</div>
								{/if}
							</div>
						</div>
					</div>
				{:else}
					<p class="text-gray-500">No results found.</p>
				{/each}
			</div>

			<div class="mt-6 flex justify-end space-x-3 border-t pt-4 dark:border-gray-700">
				<button
					on:click={close}
					class="rounded-lg bg-gray-200 px-5 py-2.5 text-sm font-medium text-gray-800 hover:bg-gray-300 dark:bg-gray-600 dark:text-white dark:hover:bg-gray-500"
				>
					{viewOnly ? 'Close' : 'Cancel'}
				</button>
				
				{#if !viewOnly}
					<button
						on:click={handleConfirm}
						class="rounded-lg bg-sky-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-sky-700"
					>
						Add to Message ({selectedIndices.length} selected)
					</button>
				{/if}
			</div>
		</div>
	</div>
{/if}