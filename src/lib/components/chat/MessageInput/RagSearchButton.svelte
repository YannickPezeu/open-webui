<script lang="ts">
	import { DropdownMenu } from 'bits-ui';
	import { flyAndScale } from '$lib/utils/transitions';
	import { getContext, onMount } from 'svelte';
	import { writable } from 'svelte/store';
	import Search from '../../icons/Search.svelte';
	import Dropdown from '../../common/Dropdown.svelte';
	import Tooltip from '../../common/Tooltip.svelte';
	import { getAvailableLibraries } from '$lib/apis/manual_rag';
	import { user } from '$lib/stores';

	const i18n = getContext('i18n');

	export let ragSearchEnabled = false;
	export let ragExpertMode = false;
	export let selectedLibraryId = '';
	export let onClose: Function = () => {};

	let show = false;
	let libraries = [];
	let loadingLibraries = false;

	onMount(async () => {
		await loadLibraries();
	});

	const loadLibraries = async () => {
		loadingLibraries = true;
		try {
			const response = await getAvailableLibraries($user.token);
			if (response && response.libraries) {
				libraries = response.libraries;
				// Sélectionner automatiquement la première bibliothèque si aucune n'est sélectionnée
				if (!selectedLibraryId && libraries.length > 0) {
					selectedLibraryId = libraries[0].library_id;
				}
			}
		} catch (error) {
			console.error('Error loading libraries:', error);
		} finally {
			loadingLibraries = false;
		}
	};

	const selectLibrary = (libraryId: string) => {
		selectedLibraryId = libraryId;
	};

	$: selectedLibrary = libraries.find(lib => lib.library_id === selectedLibraryId);
</script>

<div class="flex items-center gap-0.5">
	<!-- Bouton principal toggle ON/OFF -->
	<Tooltip content={ragSearchEnabled ? $i18n.t('RAG search enabled') : $i18n.t('RAG search disabled')} placement="top">
		<button
			on:click|preventDefault={() => {
				ragSearchEnabled = !ragSearchEnabled;
				if (!ragSearchEnabled) {
					ragExpertMode = false;
					show = false;
				}
			}}
			type="button"
			class="px-2 @xl:pl-2.5 @xl:pr-1.5 py-2 flex gap-1.5 items-center text-sm rounded-l-full transition-colors duration-300 focus:outline-hidden overflow-hidden hover:bg-gray-50 dark:hover:bg-gray-800 {ragSearchEnabled
				? 'text-sky-500 dark:text-sky-300 bg-sky-50 dark:bg-sky-200/5'
				: 'bg-transparent text-gray-600 dark:text-gray-300'}"
		>
			<Search className="size-4" strokeWidth="1.75" />
			<span
				class="hidden @xl:block whitespace-nowrap overflow-hidden text-ellipsis leading-none"
			>
				{$i18n.t('RAG Search')}
			</span>
		</button>
	</Tooltip>

	<!-- Bouton dropdown pour les options (toujours visible) -->
	<Dropdown
		bind:show
		on:change={(e) => {
			if (e.detail === false) {
				onClose();
			}
		}}
	>
		<Tooltip content={$i18n.t('RAG options')} placement="top">
			<button
				type="button"
				class="pr-2 @xl:pr-2.5 py-2 flex items-center text-sm rounded-r-full transition-colors duration-300 focus:outline-hidden hover:bg-gray-50 dark:hover:bg-gray-800 {ragSearchEnabled
					? 'text-sky-500 dark:text-sky-300 bg-sky-50 dark:bg-sky-200/5'
					: 'bg-transparent text-gray-600 dark:text-gray-300'}"
			>
				<svg 
					xmlns="http://www.w3.org/2000/svg" 
					viewBox="0 0 20 20" 
					fill="currentColor" 
					class="size-4"
				>
					<path fill-rule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clip-rule="evenodd" />
				</svg>
			</button>
		</Tooltip>

		<div slot="content">
			<DropdownMenu.Content
				class="w-full max-w-[280px] rounded-xl px-1 py-1.5 border-gray-300/30 dark:border-gray-700/50 z-50 bg-white dark:bg-gray-850 dark:text-white shadow-xl"
				sideOffset={8}
				side="top"
				align="center"
				transition={flyAndScale}
			>
				<!-- Section: Sélection de bibliothèque -->
				<div class="px-2.5 py-2 border-b border-gray-200 dark:border-gray-700">
					<span class="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">
						{$i18n.t('Library')}
					</span>
				</div>

				<div class="max-h-48 overflow-y-auto">
					{#if loadingLibraries}
						<div class="flex items-center justify-center py-4">
							<div class="animate-spin rounded-full h-5 w-5 border-b-2 border-sky-500"></div>
						</div>
					{:else if libraries.length === 0}
						<div class="px-3 py-3 text-sm text-gray-500 dark:text-gray-400 text-center">
							{$i18n.t('No libraries available')}
						</div>
					{:else}
						{#each libraries as library}
							<button
								type="button"
								on:click|stopPropagation={() => selectLibrary(library.library_id)}
								class="w-full text-left px-3 py-2.5 hover:bg-gray-50 dark:hover:bg-gray-800 rounded-lg transition flex items-center gap-2.5 group"
							>
								<div class="flex items-center justify-center w-5 h-5 flex-shrink-0">
									{#if selectedLibraryId === library.library_id}
										<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="size-5 text-sky-500">
											<path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z" clip-rule="evenodd" />
										</svg>
									{:else}
										<div class="w-4 h-4 rounded-full border-2 border-gray-300 dark:border-gray-600 group-hover:border-sky-400"></div>
									{/if}
								</div>
								<div class="flex-1 min-w-0">
									<div class="text-sm font-medium text-gray-900 dark:text-white truncate">
										{library.library_name}
									</div>
									{#if library.is_public}
										<div class="text-xs text-gray-500 dark:text-gray-400">
											{$i18n.t('Public')}
										</div>
									{/if}
								</div>
							</button>
						{/each}
					{/if}
				</div>

				<!-- Séparateur -->
				<div class="my-1 border-t border-gray-200 dark:border-gray-700"></div>

				<!-- Section: Expert Mode -->
				<label class="flex items-center gap-2.5 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 rounded-lg px-3 py-2.5 transition">
					<input
						type="checkbox"
						bind:checked={ragExpertMode}
						on:click|stopPropagation
						class="size-4 rounded border-gray-300 text-sky-600 focus:ring-sky-500 flex-shrink-0"
					/>
					<div class="flex flex-col gap-0.5">
						<span class="text-sm font-medium text-gray-900 dark:text-white whitespace-nowrap">
							{$i18n.t('Expert Mode')}
						</span>
						<span class="text-xs text-gray-500 dark:text-gray-400 leading-tight">
							{$i18n.t('Manually select sources')}
						</span>
					</div>
				</label>
			</DropdownMenu.Content>
		</div>
	</Dropdown>
</div>