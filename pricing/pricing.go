// Package pricing computes the cost of echo calls from token usage.
//
// Prices come from a snapshot of models.dev (https://models.dev) embedded in
// the package, plus hand-maintained prices for providers models.dev does not
// cover (Voyage AI, TypeSafe). A fresh table can be downloaded with Fetch.
//
//	cost, ok := pricing.Default().Cost(resp.Usage)
//
// or let the client fill Usage.CostUSD on every response:
//
//	client, _ := echo.NewCommonClient(nil, echo.WithPricer(pricing.Default()))
//
// All prices are USD per 1M tokens. Prices are estimates: free tiers, batch
// discounts and account-specific rates are not taken into account.
package pricing

//go:generate go run ./internal/gen -o prices.json

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"

	"github.com/mkozhukh/echo"
)

// SourceURL is the models.dev dataset downloaded by Fetch.
const SourceURL = "https://models.dev/api.json"

// prices.json: models.dev snapshot for echo's LLM providers, see internal/gen.
//
//go:embed prices.json
var snapshotData []byte

// extra.json: providers missing from models.dev, maintained by hand.
//   - voyage: https://docs.voyageai.com/docs/pricing
//   - typesafe: https://vercel.com/ai-gateway/models/providers/typesafe-ai
//     (input price only; no output price is published)
//
//go:embed extra.json
var extraData []byte

// Rates are prices in USD per 1M tokens.
// Zero CacheRead/CacheWrite mean "billed as Input"; zero Reasoning means
// reasoning tokens are billed as Output.
type Rates struct {
	Input      float64
	Output     float64
	CacheRead  float64
	CacheWrite float64
	Reasoning  float64
}

// Tier applies its rates to the whole call when the input exceeds Above tokens
// (e.g. Gemini "context over 200k" pricing).
type Tier struct {
	Above int
	Rates
}

// Price is the pricing of a single model.
type Price struct {
	Rates
	Tiers []Tier // sorted by Above, ascending
}

// rates returns the rates applicable to a call with the given input size
func (p Price) rates(inputTokens int) Rates {
	r := p.Rates
	for _, t := range p.Tiers {
		if inputTokens > t.Above {
			r = t.Rates
		}
	}
	return r
}

// Cost is the cost breakdown of a call in USD.
type Cost struct {
	Input       float64 // uncached input
	CachedInput float64 // input read from cache
	CacheWrite  float64 // input written to cache
	Output      float64 // output, excluding separately priced reasoning
	Reasoning   float64 // reasoning, when the model prices it separately
	Total       float64
	Reported    bool // Total is the provider-reported cost (Usage.CostUSD); no breakdown
}

// Prices is a price table keyed by echo model names ("provider/model").
// It is safe for concurrent use.
type Prices struct {
	mu     sync.RWMutex
	models map[string]map[string]Price // provider -> model -> price
}

var (
	defaultOnce   sync.Once
	defaultPrices *Prices
)

// Default returns the built-in price table (models.dev snapshot plus Voyage
// and TypeSafe prices). The table is shared by the whole process: changes made
// with Set are visible to every user of Default. Use Clone for a private copy.
func Default() *Prices {
	defaultOnce.Do(func() {
		p, err := builtin()
		if err != nil {
			panic("pricing: invalid embedded price data: " + err.Error())
		}
		defaultPrices = p
	})
	return defaultPrices
}

func builtin() (*Prices, error) {
	p, err := parse(snapshotData)
	if err != nil {
		return nil, err
	}
	extra, err := parse(extraData)
	if err != nil {
		return nil, err
	}
	p.merge(extra)
	return p, nil
}

// Fetch downloads the current models.dev dataset and adds the built-in prices
// for providers models.dev does not cover. On failure it returns a copy of the
// built-in table together with the error, so the result is always usable.
func Fetch(ctx context.Context) (*Prices, error) {
	p, err := fetch(ctx, SourceURL)
	if err != nil {
		return Default().Clone(), err
	}
	return p, nil
}

func fetch(ctx context.Context, url string) (*Prices, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("pricing: fetch %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("pricing: fetch %s: %s", url, resp.Status)
	}

	p, err := Load(resp.Body)
	if err != nil {
		return nil, err
	}
	extra, _ := parse(extraData)
	p.addMissing(extra)
	return p, nil
}

// Load reads a price table in models.dev api.json format: the full dataset,
// or a table previously written with Save.
func Load(r io.Reader) (*Prices, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	return parse(data)
}

// Save writes the table in models.dev api.json format (cost fields only),
// readable by Load. Use it to cache a Fetch result.
func (p *Prices) Save(w io.Writer) error {
	p.mu.RLock()
	out := make(map[string]providerJSON, len(p.models))
	for prov, models := range p.models {
		pj := providerJSON{Models: make(map[string]modelJSON, len(models))}
		for id, price := range models {
			pj.Models[id] = modelJSON{Cost: toCostJSON(price)}
		}
		out[prov] = pj
	}
	p.mu.RUnlock()

	enc := json.NewEncoder(w)
	enc.SetIndent("", " ")
	return enc.Encode(out)
}

// Clone returns an independent copy of the table.
func (p *Prices) Clone() *Prices {
	p.mu.RLock()
	defer p.mu.RUnlock()
	c := &Prices{models: make(map[string]map[string]Price, len(p.models))}
	for prov, models := range p.models {
		m := make(map[string]Price, len(models))
		for id, price := range models {
			m[id] = price
		}
		c.models[prov] = m
	}
	return c
}

// Set adds or replaces the price of a model ("provider/model").
func (p *Prices) Set(model string, price Price) {
	prov, id := splitModel(model)
	sort.Slice(price.Tiers, func(i, j int) bool { return price.Tiers[i].Above < price.Tiers[j].Above })

	p.mu.Lock()
	defer p.mu.Unlock()
	if p.models[prov] == nil {
		p.models[prov] = map[string]Price{}
	}
	p.models[prov][id] = price
}

// Get returns the price of a model ("provider/model").
func (p *Prices) Get(model string) (Price, bool) {
	prov, id := splitModel(model)
	p.mu.RLock()
	defer p.mu.RUnlock()
	price, ok := p.models[prov][id]
	return price, ok
}

// Cost computes the cost of a call. A cost reported by the provider
// (Usage.CostUSD) is returned as is, with Reported set. ok is false when u is
// nil or no price is known for u.Model.
func (p *Prices) Cost(u *echo.Usage) (Cost, bool) {
	if u == nil {
		return Cost{}, false
	}
	if u.CostUSD != nil {
		return Cost{Total: *u.CostUSD, Reported: true}, true
	}
	price, ok := p.Get(u.Model)
	if !ok {
		return Cost{}, false
	}
	r := price.rates(u.InputTokens)

	cacheRead, cacheWrite := r.CacheRead, r.CacheWrite
	if cacheRead == 0 {
		cacheRead = r.Input
	}
	if cacheWrite == 0 {
		cacheWrite = r.Input
	}

	fresh := max(u.InputTokens-u.CachedInputTokens-u.CacheWriteTokens, 0)
	c := Cost{
		Input:       perM(fresh, r.Input),
		CachedInput: perM(u.CachedInputTokens, cacheRead),
		CacheWrite:  perM(u.CacheWriteTokens, cacheWrite),
	}
	if r.Reasoning > 0 {
		c.Reasoning = perM(u.ReasoningTokens, r.Reasoning)
		c.Output = perM(max(u.OutputTokens-u.ReasoningTokens, 0), r.Output)
	} else {
		c.Output = perM(u.OutputTokens, r.Output)
	}
	c.Total = c.Input + c.CachedInput + c.CacheWrite + c.Output + c.Reasoning
	return c, true
}

// USD returns the total cost of a call. It implements echo.Pricer, so a table
// can be passed to echo.WithPricer.
func (p *Prices) USD(u echo.Usage) (float64, bool) {
	c, ok := p.Cost(&u)
	return c.Total, ok
}

// Enrich sets u.CostUSD when it is not already set and a price is known.
// It reports whether u.CostUSD is set afterwards.
func (p *Prices) Enrich(u *echo.Usage) bool {
	if u == nil {
		return false
	}
	if u.CostUSD != nil {
		return true
	}
	usd, ok := p.USD(*u)
	if ok {
		u.CostUSD = &usd
	}
	return ok
}

var _ echo.Pricer = (*Prices)(nil)

func perM(tokens int, usdPerM float64) float64 {
	return float64(tokens) * usdPerM / 1e6
}

// splitModel splits "provider/model"; the model part may contain slashes (OpenRouter)
func splitModel(model string) (string, string) {
	prov, id, _ := strings.Cut(model, "/")
	return prov, id
}

// merge overwrites prices of p with the ones of other
func (p *Prices) merge(other *Prices) {
	for prov, models := range other.models {
		if p.models[prov] == nil {
			p.models[prov] = map[string]Price{}
		}
		for id, price := range models {
			p.models[prov][id] = price
		}
	}
}

// addMissing copies prices of other that p does not have
func (p *Prices) addMissing(other *Prices) {
	for prov, models := range other.models {
		if p.models[prov] == nil {
			p.models[prov] = map[string]Price{}
		}
		for id, price := range models {
			if _, ok := p.models[prov][id]; !ok {
				p.models[prov][id] = price
			}
		}
	}
}

// models.dev api.json format (only the fields used here)

type providerJSON struct {
	Models map[string]modelJSON `json:"models"`
}

type modelJSON struct {
	Cost *costJSON `json:"cost,omitempty"`
}

type ratesJSON struct {
	Input      *float64 `json:"input,omitempty"`
	Output     *float64 `json:"output,omitempty"`
	CacheRead  *float64 `json:"cache_read,omitempty"`
	CacheWrite *float64 `json:"cache_write,omitempty"`
	Reasoning  *float64 `json:"reasoning,omitempty"`
}

type tierJSON struct {
	ratesJSON
	Tier struct {
		Type string `json:"type"`
		Size int    `json:"size"`
	} `json:"tier"`
}

type costJSON struct {
	ratesJSON
	Tiers           []tierJSON `json:"tiers,omitempty"`
	ContextOver200k *ratesJSON `json:"context_over_200k,omitempty"`
}

func (r ratesJSON) rates() Rates {
	v := func(f *float64) float64 {
		if f == nil {
			return 0
		}
		return *f
	}
	return Rates{Input: v(r.Input), Output: v(r.Output), CacheRead: v(r.CacheRead), CacheWrite: v(r.CacheWrite), Reasoning: v(r.Reasoning)}
}

func toRatesJSON(r Rates) ratesJSON {
	ptr := func(f float64) *float64 {
		if f == 0 {
			return nil
		}
		return &f
	}
	in, out := r.Input, r.Output
	return ratesJSON{Input: &in, Output: &out, CacheRead: ptr(r.CacheRead), CacheWrite: ptr(r.CacheWrite), Reasoning: ptr(r.Reasoning)}
}

func toCostJSON(p Price) *costJSON {
	c := &costJSON{ratesJSON: toRatesJSON(p.Rates)}
	for _, t := range p.Tiers {
		tj := tierJSON{ratesJSON: toRatesJSON(t.Rates)}
		tj.Tier.Type = "context"
		tj.Tier.Size = t.Above
		c.Tiers = append(c.Tiers, tj)
	}
	return c
}

func parse(data []byte) (*Prices, error) {
	var raw map[string]providerJSON
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("pricing: invalid price data: %w", err)
	}

	p := &Prices{models: make(map[string]map[string]Price, len(raw))}
	for prov, pj := range raw {
		models := make(map[string]Price, len(pj.Models))
		for id, m := range pj.Models {
			if m.Cost == nil || m.Cost.Input == nil {
				continue
			}
			price := Price{Rates: m.Cost.rates()}
			for _, t := range m.Cost.Tiers {
				if t.Tier.Type == "context" && t.Tier.Size > 0 {
					price.Tiers = append(price.Tiers, Tier{Above: t.Tier.Size, Rates: t.rates()})
				}
			}
			if len(price.Tiers) == 0 && m.Cost.ContextOver200k != nil {
				price.Tiers = []Tier{{Above: 200_000, Rates: m.Cost.ContextOver200k.rates()}}
			}
			sort.Slice(price.Tiers, func(i, j int) bool { return price.Tiers[i].Above < price.Tiers[j].Above })
			models[id] = price
		}
		if len(models) > 0 {
			p.models[prov] = models
		}
	}
	return p, nil
}
