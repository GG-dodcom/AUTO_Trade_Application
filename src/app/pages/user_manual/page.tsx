import React from "react";

const UserManaual: React.FC = () => {
	return (
		<div className="max-w-4xl mx-auto p-6 bg-white shadow-lg rounded-lg">
			<h1 className="text-3xl font-bold text-gray-800 mb-6 text-center">
				User Manual: AUTO Trade Application
			</h1>
			<p className="text-gray-600 mb-6">
				This user manual provides step-by-step instructions to set up and
				operate the AUTO Trade application for automated trading on the XAUUSD
				market using MetaTrader 5 (MT5). Follow these steps carefully to ensure
				a smooth experience. Each step includes references to figures for visual
				guidance and troubleshooting tips where applicable.
			</p>

			{/* Prerequisites */}
			<h2 className="text-2xl font-semibold text-gray-800 mb-4">
				Prerequisites
			</h2>
			<ul className="list-disc list-inside text-gray-600 mb-6">
				<li>A stable internet connection.</li>
				<li>A Windows PC (for running Auto Trade.exe).</li>
				<li>
					Basic familiarity with trading platforms and account management.
				</li>
			</ul>

			{/* Step-by-Step Instructions */}
			<h2 className="text-2xl font-semibold text-gray-800 mb-4">
				Step-by-Step Instructions
			</h2>

			{/* Step 1: Install MetaTrader5 */}
			<h3 className="text-xl font-semibold text-gray-700 mb-3">
				Step 1: Install MetaTrader5
			</h3>
			<ol className="list-decimal list-inside text-gray-600 mb-4">
				<li>
					Visit{" "}
					<a
						href="https://www.metatrader5.com/en/download"
						className="text-blue-600 hover:underline"
					>
						https://www.metatrader5.com/en/download
					</a>
					.
				</li>
				<li>Download the MetaTrader 5 platform for Windows.</li>
				<li>
					Run the installer and follow the on-screen instructions to complete
					the installation.
				</li>
			</ol>
			<img
				src="/media/image1.png"
				alt="Download MetaTrader5 platform"
				className="w-full max-w-xl mx-auto mb-4 rounded-lg shadow-md"
			/>
			<p className="text-gray-500 italic text-sm mb-4 text-center">
				Figure 1: Download MetaTrader5 platform.
			</p>
			<p className="text-gray-600 bg-gray-100 p-3 rounded-md mb-6">
				<span className="font-semibold">Troubleshooting Tip:</span> If the
				download fails, check your internet connection or try a different
				browser. Ensure your system meets MT5’s minimum requirements (Windows 7
				or later).
			</p>

			{/* Step 2: Register a Demo Account */}
			<h3 className="text-xl font-semibold text-gray-700 mb-3">
				Step 2: Register a Demo Account
			</h3>
			<ol className="list-decimal list-inside text-gray-600 mb-4">
				<li>
					Navigate to{" "}
					<a
						href="https://en.octafxmy.net/"
						className="text-blue-600 hover:underline"
					>
						https://en.octafxmy.net/
					</a>
					.
				</li>
				<li>Select the option to create a demo account for MetaTrader 5.</li>
				<li>
					Fill in the required details (e.g., name, email, and preferred
					currency).
				</li>
				<li>
					Save the account credentials (account number, password, and server)
					provided upon registration.
				</li>
			</ol>
			<img
				src="/media/image2.png"
				alt="Create a demo account for MetaTrader5 platform"
				className="w-full max-w-xl mx-auto mb-4 rounded-lg shadow-md"
			/>
			<img
				src="/media/image3.png"
				alt="Create a demo account for MetaTrader5 platform"
				className="w-full max-w-xl mx-auto mb-4 rounded-lg shadow-md"
			/>
			<p className="text-gray-500 italic text-sm mb-4 text-center">
				Figure 2: Create a demo account for MetaTrader5 platform.
			</p>
			<img
				src="/media/image4.png"
				alt="Save account credentials"
				className="w-full max-w-xl mx-auto mb-4 rounded-lg shadow-md"
			/>
			<p className="text-gray-500 italic text-sm mb-4 text-center">
				Figure 3: Save this account number, server, and password to login to
				your MT5 broker account.
			</p>
			<p className="text-gray-600 bg-gray-100 p-3 rounded-md mb-6">
				<span className="font-semibold">Note:</span> A demo account allows you
				to test the trading system without financial risk. Keep your credentials
				secure for future use.
			</p>

			{/* Step 3: Login to MetaTrader5 */}
			<h3 className="text-xl font-semibold text-gray-700 mb-3">
				Step 3: Login to MetaTrader5
			</h3>
			<ol className="list-decimal list-inside text-gray-600 mb-4">
				<li>Launch the MetaTrader5 application.</li>
				<li>
					Enter your demo account credentials (account number, password, and
					server).
				</li>
				<li>
					Verify the login by checking the connection status in the bottom-right
					corner of MT5 (should display “Connected”).
				</li>
			</ol>
			<div className="flex justify-center space-x-4 mb-4">
				<img
					src="/media/image5.png"
					alt="Login to MetaTrader5"
					className="w-1/2 max-w-xs rounded-lg shadow-md"
				/>
				<img
					src="/media/image6.png"
					alt="Login to MetaTrader5"
					className="w-1/2 max-w-xs rounded-lg shadow-md"
				/>
			</div>
			<p className="text-gray-500 italic text-sm mb-4 text-center">
				Figure 4: Login to MetaTrader5.
			</p>
			<p className="text-gray-600 bg-gray-100 p-3 rounded-md mb-6">
				<span className="font-semibold">Troubleshooting Tip:</span> If login
				fails, ensure the server name matches exactly as provided by OctaFX.
				Contact your broker’s support if issues persist.
			</p>

			{/* Step 4: Enable Algorithmic Trading */}
			<h3 className="text-xl font-semibold text-gray-700 mb-3">
				Step 4: Enable Algorithmic Trading
			</h3>
			<ol className="list-decimal list-inside text-gray-600 mb-4">
				<li>
					In MetaTrader 5, locate the “Algo Trading” button in the toolbar.
				</li>
				<li>
					Click the button to enable algorithmic trading (the button will turn
					green when activated).
				</li>
			</ol>
			<img
				src="/media/image7.png"
				alt="Enable algorithm trade on MetaTrader5"
				className="w-full max-w-2xl mx-auto mb-4 rounded-lg shadow-md"
			/>
			<img
				src="/media/image8.png"
				alt="Enable algorithm trade on MetaTrader5"
				className="w-full max-w-2xl mx-auto mb-4 rounded-lg shadow-md"
			/>
			<p className="text-gray-500 italic text-sm mb-4 text-center">
				Figure 5: Enable algorithm trade on MetaTrader5.
			</p>
			<p className="text-gray-600 bg-yellow-100 p-3 rounded-md mb-4">
				<span className="font-semibold">Important:</span> Algorithmic trading
				must be enabled for the AUTO Trade application to execute trades. If the
				run model controller button remains gray, check MT5’s toolbar to enable
				the ‘Algo Trading’.
			</p>
			<img
				src="/media/image9.png"
				alt="Run model controller grayed out"
				className="w-full max-w-[15rem] mx-auto mb-4 rounded-lg shadow-md"
			/>
			<p className="text-gray-500 italic text-sm mb-4 text-center">
				Figure 6: Run model controller will show in gray if ‘Algo Trading’ is
				disabled.
			</p>

			{/* Step 5: Install and Launch AUTO Trade Application */}
			<h3 className="text-xl font-semibold text-gray-700 mb-3">
				Step 5: Install and Launch AUTO Trade Application
			</h3>
			<ol className="list-decimal list-inside text-gray-600 mb-4">
				<li>
					Download the AUTO Trade application zip file from the provided source.
				</li>
				<li>Extract the zip file to a preferred location on your computer.</li>
				<li>
					Navigate to the extracted folder and double-click Auto Trade.exe to
					launch the application.
				</li>
			</ol>
			<img
				src="/media/image10.png"
				alt="Unzip folder"
				className="w-full max-w-xs mx-auto mb-4 rounded-lg shadow-md"
			/>
			<p className="text-gray-500 italic text-sm mb-4 text-center">
				Figure 7: Unzip folder.
			</p>
			<img
				src="/media/image11.png"
				alt="Launch Auto Trade application"
				className="w-full max-w-xl mx-auto mb-4 rounded-lg shadow-md"
			/>
			<p className="text-gray-500 italic text-sm mb-4 text-center">
				Figure 8: Double click on ‘Auto Trade’ application to open it.
			</p>

			{/* Step 6: Sign Up for AUTO Trade */}
			<h3 className="text-xl font-semibold text-gray-700 mb-3">
				Step 6: Sign Up for AUTO Trade
			</h3>
			<ol className="list-decimal list-inside text-gray-600 mb-4">
				<li>On the application’s startup screen, click “Sign Up.”</li>
				<li>Enter a valid email address and a password.</li>
				<li>Submit the form to create your account.</li>
			</ol>
			<img
				src="/media/image12.png"
				alt="Frontend signup page"
				className="w-full max-w-xl mx-auto mb-4 rounded-lg shadow-md"
			/>
			<p className="text-gray-500 italic text-sm mb-4 text-center">
				Figure 9: Frontend signup page.
			</p>
			<p className="text-gray-600 bg-gray-100 p-3 rounded-md mb-6">
				<span className="font-semibold">Note:</span> If you encounter an error
				(e.g., “Email already in use”), try a different email.
			</p>

			{/* Step 7: Login to AUTO Trade */}
			<h3 className="text-xl font-semibold text-gray-700 mb-3">
				Step 7: Login to AUTO Trade
			</h3>
			<ol className="list-decimal list-inside text-gray-600 mb-4">
				<li>Return to the login page.</li>
				<li>Enter your registered email and password.</li>
				<li>Click “Login” to access the application.</li>
			</ol>
			<img
				src="/media/image13.png"
				alt="Frontend login page"
				className="w-full max-w-xl mx-auto mb-4 rounded-lg shadow-md"
			/>
			<p className="text-gray-500 italic text-sm mb-4 text-center">
				Figure 10: Frontend login page.
			</p>

			{/* Step 8: Connect AUTO Trade to MetaTrader5 */}
			<h3 className="text-xl font-semibold text-gray-700 mb-3">
				Step 8: Connect AUTO Trade to MetaTrader5
			</h3>
			<ol className="list-decimal list-inside text-gray-600 mb-4">
				<li>
					In the AUTO Trade application, navigate to the MT5 account setup
					section.
				</li>
				<li>
					Enter your MT5 demo account details (account number, password, and
					server).
				</li>
				<li>
					Optionally, check “Save Password” to securely store your credentials
					in the application’s database for automatic retrieval in future
					sessions.
				</li>
				<li>
					Submit the form. If successful, MetaTrader5 will automatically open,
					confirming the connection.
				</li>
			</ol>
			<img
				src="/media/image14.png"
				alt="Fill up the detail to initial MetaTrader5 platform"
				className="w-full max-w-xl mx-auto mb-4 rounded-lg shadow-md"
			/>
			<p className="text-gray-500 italic text-sm mb-4 text-center">
				Figure 11: Fill up the detail to initial MetaTrader5 platform.
			</p>
			<img
				src="/media/image15.png"
				alt="Save password option"
				className="w-full max-w-xl mx-auto mb-4 rounded-lg shadow-md"
			/>
			<p className="text-gray-500 italic text-sm mb-4 text-center">
				Figure 12: If ‘Save password’ checked, it will auto retrieve MT5 account
				detail next time.
			</p>
			<img
				src="/media/image16.png"
				alt="MT5 auto-opens after successful login"
				className="w-full max-w-xl mx-auto mb-4 rounded-lg shadow-md"
			/>
			<p className="text-gray-500 italic text-sm mb-4 text-center">
				Figure 13: MT5 auto-opens after successful login to MT5 broker account.
			</p>

			{/* Step 9: Run the AI Trading Model */}
			<h3 className="text-xl font-semibold text-gray-700 mb-3">
				Step 9: Run the AI Trading Model
			</h3>
			<ol className="list-decimal list-inside text-gray-600 mb-4">
				<li>
					In the AUTO Trade dashboard, locate the “Play” button for the AI
					auto-trade model.
				</li>
				<li>
					Click the “Play” button to start the model. The status will update to
					“Running.”
				</li>
				<li>To stop the model, click the “Stop” button.</li>
			</ol>
			<div className="flex justify-center space-x-4 mb-4">
				<img
					src="/media/image17.png"
					alt="Run model controller stop & start"
					className="w-1/2 max-w-[15rem] rounded-lg shadow-md"
				/>
				<img
					src="/media/image18.png"
					alt="Run model controller stop & start"
					className="w-1/2 max-w-[15rem] rounded-lg shadow-md"
				/>
			</div>
			<p className="text-gray-500 italic text-sm mb-4 text-center">
				Figure 14: Run model controller stop & start.
			</p>
			<p className="text-gray-600 bg-gray-100 p-3 rounded-md mb-6">
				<span className="font-semibold">Note:</span> The AI model uses a
				PPO-based reinforcement learning algorithm to execute trades. Monitor
				its performance to ensure it aligns with your trading goals.
			</p>

			{/* Step 10: Monitor and Analyze Trading Performance */}
			<h3 className="text-xl font-semibold text-gray-700 mb-3">
				Step 10: Monitor and Analyze Trading Performance
			</h3>
			<ol className="list-decimal list-inside text-gray-600 mb-4">
				<li>
					Navigate to the “Backtest Report” to review the AI model’s historical
					performance metrics (e.g., cumulative return, drawdown).
				</li>
				<li>
					Use the “Dashboard” to view real-time trade updates, including charts
					and notifications.
				</li>
				<li>
					Access the “Trading Journal” to track individual trade details, such
					as entry/exit times and profit/loss.
				</li>
				<li>
					Visit the “Monthly Report” for aggregated performance statistics by
					month.
				</li>
				<li>In the “Settings” page, customize trading parameters:</li>
				<ul className="list-disc list-inside ml-6">
					<li>
						Symbol: Specific XAUUSD symbol name assigned by the broker account.
					</li>
					<li>Volume: Set the trade size (e.g., 0.1 lots).</li>
					<li>Stop Loss: Define the maximum loss per trade in points.</li>
					<li>Take Profit: Set the target profit per trade in points.</li>
				</ul>
			</ol>
			<img
				src="/media/image19.png"
				alt="Frontend backtest report page"
				className="w-full max-w-xl mx-auto mb-4 rounded-lg shadow-md"
			/>
			<p className="text-gray-500 italic text-sm mb-4 text-center">
				Figure 15: Frontend backtest report page.
			</p>
			<img
				src="/media/image20.png"
				alt="Frontend dashboard pages"
				className="w-full max-w-xl mx-auto mb-4 rounded-lg shadow-md"
			/>
			<p className="text-gray-500 italic text-sm mb-4 text-center">
				Figure 16: Frontend dashboard pages.
			</p>
			<img
				src="/media/image21.png"
				alt="Frontend trading journal page"
				className="w-full max-w-xl mx-auto mb-4 rounded-lg shadow-md"
			/>
			<p className="text-gray-500 italic text-sm mb-4 text-center">
				Figure 17: Frontend trading journal page.
			</p>
			<img
				src="/media/image22.png"
				alt="Frontend monthly report page"
				className="w-full max-w-xl mx-auto mb-4 rounded-lg shadow-md"
			/>
			<p className="text-gray-500 italic text-sm mb-4 text-center">
				Figure 18: Frontend monthly report page.
			</p>
			<img
				src="/media/image23.png"
				alt="Frontend setting page"
				className="w-full max-w-xl mx-auto mb-4 rounded-lg shadow-md"
			/>
			<p className="text-gray-500 italic text-sm mb-4 text-center">
				Figure 19: Frontend setting page.
			</p>
		</div>
	);
};

export default UserManaual;
