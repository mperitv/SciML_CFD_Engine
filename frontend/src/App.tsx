import { FormEvent, useEffect, useState } from 'react'

interface SimulationResponse {
  task_id: string
  status: string
  warning?: string | null
}

interface StatusResponse {
  task_id: string
  status: string
  created_at: number
  started_at: number | null
  finished_at: number | null
  error: string | null
  warning: string | null
  image_path: string | null
  model_path: string | null
}

function App() {
  const [reynoldsNumber, setReynoldsNumber] = useState(100.0)
  const [inletVelocity, setInletVelocity] = useState(1.0)
  const [radius, setRadius] = useState(0.5)
  const [length, setLength] = useState(3.0)
  const [epochs, setEpochs] = useState(1000)
  const [batchInterior, setBatchInterior] = useState(1000)
  const [batchBoundary, setBatchBoundary] = useState(200)
  const [lbfgsEpochs, setLbfgsEpochs] = useState(100)
  const [taskId, setTaskId] = useState<string | null>(null)
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [warning, setWarning] = useState<string | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const submitSimulation = async (event: FormEvent) => {
    event.preventDefault()
    setErrorMessage(null)
    setStatus(null)
    setTaskId(null)
    setWarning(null)

    const payload = {
      reynolds_number: reynoldsNumber,
      inlet_velocity: inletVelocity,
      radius,
      length,
      epochs,
      batch_interior: batchInterior,
      batch_boundary: batchBoundary,
      lbfgs_epochs: lbfgsEpochs,
    }

    try {
      const response = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        const body = await response.json()
        throw new Error(body.detail || 'Failed to start simulation')
      }

      const result: SimulationResponse = await response.json()
      setTaskId(result.task_id)
      setStatus({
        task_id: result.task_id,
        status: result.status,
        created_at: Date.now() / 1000,
        started_at: null,
        finished_at: null,
        error: null,
        warning: result.warning ?? null,
        image_path: null,
        model_path: null,
      })
      setWarning(result.warning ?? null)
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Unknown error')
    }
  }

  const fetchStatus = async () => {
    if (!taskId) {
      return
    }

    try {
      const response = await fetch(`/api/status/${taskId}`)
      if (!response.ok) {
        throw new Error('Unable to fetch status')
      }
      const result: StatusResponse = await response.json()
      setStatus(result)
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Status request failed')
    }
  }

  useEffect(() => {
    if (!taskId) {
      return
    }

    const interval = window.setInterval(() => {
      fetchStatus()
    }, 3000)

    return () => window.clearInterval(interval)
  }, [taskId])

  return (
    <div className="app-shell">
      <header>
        <h1>SciML CFD Engine</h1>
        <p>Launch a PINN pipe-flow simulation and inspect the visual result.</p>
      </header>

      <main>
        <form onSubmit={submitSimulation} className="sim-form">
          <label>
            Reynolds number
            <input
              type="number"
              min="1"
              step="1"
              value={reynoldsNumber}
              onChange={(event) => setReynoldsNumber(Number(event.target.value))}
            />
          </label>

          <label>
            Inlet velocity
            <input
              type="number"
              min="0.01"
              step="0.01"
              value={inletVelocity}
              onChange={(event) => setInletVelocity(Number(event.target.value))}
            />
          </label>

          <label>
            Pipe radius
            <input
              type="number"
              min="0.01"
              step="0.01"
              value={radius}
              onChange={(event) => setRadius(Number(event.target.value))}
            />
          </label>

          <label>
            Pipe length
            <input
              type="number"
              min="0.1"
              step="0.1"
              value={length}
              onChange={(event) => setLength(Number(event.target.value))}
            />
          </label>

          <label>
            Adam epochs
            <input
              type="number"
              min="1"
              step="1"
              value={epochs}
              onChange={(event) => setEpochs(Number(event.target.value))}
            />
          </label>

          <label>
            Interior batch size
            <input
              type="number"
              min="1"
              step="1"
              value={batchInterior}
              onChange={(event) => setBatchInterior(Number(event.target.value))}
            />
          </label>

          <label>
            Boundary batch size
            <input
              type="number"
              min="1"
              step="1"
              value={batchBoundary}
              onChange={(event) => setBatchBoundary(Number(event.target.value))}
            />
          </label>

          <label>
            L-BFGS iterations
            <input
              type="number"
              min="0"
              step="1"
              value={lbfgsEpochs}
              onChange={(event) => setLbfgsEpochs(Number(event.target.value))}
            />
          </label>

          <button type="submit">Start simulation</button>
        </form>

        <section className="status-panel">
          {errorMessage ? <div className="toast error">{errorMessage}</div> : null}
          {warning ? <div className="toast warning">{warning}</div> : null}
          {status ? (
            <div>
              <h2>Task status</h2>
              <p>
                <strong>Task ID:</strong> {status.task_id}
              </p>
              <p>
                <strong>Status:</strong> {status.status}
              </p>
              {status.error ? <p className="error-text">{status.error}</p> : null}
              {status.status === 'completed' ? (
                <div>
                  <h3>Result</h3>
                  <img src={`/api/result/${status.task_id}`} alt="Pipe flow result" />
                </div>
              ) : null}
            </div>
          ) : (
            <p>Submit the form to create a simulation task.</p>
          )}
        </section>
      </main>
    </div>
  )
}

export default App
