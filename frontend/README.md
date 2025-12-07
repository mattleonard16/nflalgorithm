# NFL Algorithm Frontend

Modern React dashboard for the NFL Algorithm betting system.

## Tech Stack

- **Framework**: Next.js 15 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS v4
- **Components**: shadcn/ui
- **Charts**: Recharts
- **Icons**: Lucide React

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn
- FastAPI backend running on port 8000

### Installation

```bash
cd frontend
npm install
```

### Environment Setup

Create a `.env.local` file in the frontend directory:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Development

```bash
# Start the development server
npm run dev

# The app will be available at http://localhost:3000
```

### Production Build

```bash
npm run build
npm start
```

## Project Structure

```
src/
├── app/                    # Next.js App Router pages
│   ├── page.tsx           # Dashboard (Live Bets)
│   ├── performance/       # Performance history
│   ├── analytics/         # Edge analytics
│   ├── system/            # System health
│   └── settings/          # User settings
├── components/
│   ├── ui/                # shadcn components
│   └── sidebar.tsx        # Main navigation
└── lib/
    ├── api.ts             # API client functions
    ├── types.ts           # TypeScript types
    └── utils.ts           # Utility functions
```

## Pages

| Route | Description |
|-------|-------------|
| `/` | Main dashboard with value bets |
| `/performance` | Historical performance and P/L charts |
| `/analytics` | Edge distribution and market analysis |
| `/system` | System health and feed freshness |
| `/settings` | User preferences and parameters |

## API Integration

The frontend connects to the FastAPI backend at `NEXT_PUBLIC_API_URL`.

### Endpoints Used

- `GET /api/meta` - Available weeks, sportsbooks, markets
- `GET /api/value-bets` - Value betting opportunities
- `GET /api/performance` - Historical performance data
- `GET /api/health` - System health status
- `GET /api/analytics/*` - Edge distribution and stats

## Development Tips

1. Start the backend first: `make api`
2. Then start the frontend: `cd frontend && npm run dev`
3. Both should be running for full functionality
