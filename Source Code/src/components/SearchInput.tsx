import { Search } from "@/assets/icons";
import React, { useState } from "react";

interface SearchInputProps {
	onSearch?: (params: { query: string; category: string }) => void;
}

const SearchInput: React.FC<SearchInputProps> = ({ onSearch }) => {
	const [query, setQuery] = useState("");
	const [selectedCategory, setSelectedCategory] = useState("All categories");
	const [dropdownOpen, setDropdownOpen] = useState(false);

	const categories = [
		"Order ID",
		"Direction",
		"Currency Pair",
		"Model",
		"Status",
	];

	const handleSearch = (e: React.FormEvent<HTMLFormElement>) => {
		e.preventDefault();
		if (onSearch) onSearch({ query, category: selectedCategory });
	};

	return (
		<form onSubmit={handleSearch} className="max-w-lg mx-auto">
			<div className="flex relative">
				{/* Dropdown Button */}
				<button
					type="button"
					onClick={() => setDropdownOpen(!dropdownOpen)}
					className="shrink-0 z-10 inline-flex items-center py-2.5 px-4 text-sm font-medium text-gray-900 bg-gray-100 border border-gray-300 rounded-s-lg hover:bg-gray-200 focus:ring-4 focus:outline-none dark:bg-gray-700 dark:hover:bg-gray-600 dark:text-white"
				>
					{selectedCategory}
					<svg
						className="w-2.5 h-2.5 ml-2"
						viewBox="0 0 10 6"
						fill="none"
						xmlns="http://www.w3.org/2000/svg"
					>
						<path
							stroke="currentColor"
							strokeLinecap="round"
							strokeLinejoin="round"
							strokeWidth="2"
							d="m1 1 4 4 4-4"
						/>
					</svg>
				</button>

				{/* Dropdown Menu */}
				{dropdownOpen && (
					<div className="absolute left-0 top-full bg-white divide-y divide-gray-100 rounded-lg shadow-lg w-44 dark:bg-gray-700 z-20">
						<ul className="py-2 text-sm text-gray-700 dark:text-gray-200">
							{categories.map((category) => (
								<li key={category}>
									<button
										type="button"
										onClick={() => {
											setSelectedCategory(category);
											setDropdownOpen(false);
										}}
										className="block w-full text-left px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-600 dark:text-white"
									>
										{category}
									</button>
								</li>
							))}
						</ul>
					</div>
				)}

				{/* Search Input */}
				<div className="relative w-full">
					<input
						type="search"
						value={query}
						onChange={(e) => setQuery(e.target.value)}
						className="block p-2.5 w-full text-sm text-gray-900 bg-gray-50 border border-gray-300 rounded-e-lg focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white dark:focus:border-blue-500"
						placeholder="Search Mockups, Logos, Design Templates..."
						required
					/>
					{/* Search Button */}
					<button
						type="submit"
						className="absolute top-0 right-0 p-2.5 text-sm font-medium h-full text-white bg-blue-700 rounded-e-lg border border-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800"
					>
						<Search className="w-4 h-4" />
					</button>
				</div>
			</div>
		</form>
	);
};

export default SearchInput;
