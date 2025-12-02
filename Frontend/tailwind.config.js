/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                orange: {
                    50: '#fff7ed',
                    100: '#ffedd5',
                    500: '#f97316',
                    600: '#ea580c',
                    700: '#c2410c',
                },
                amber: {
                    400: '#fbbf24',
                    500: '#f59e0b',
                    600: '#d97706',
                },
                red: {
                    500: '#ef4444',
                }
            },
            borderRadius: {
                '2xl': '1rem',
                '3xl': '1.5rem',
            }
        },
    },
    plugins: [],
}
