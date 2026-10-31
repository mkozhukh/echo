package echo

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
)

// Question types supported by decision calls
const (
	QuestionChoice = "choice"
	QuestionScore  = "score"
	QuestionNoul   = "noul"
)

// Questions is a set of typed questions keyed by an id of your choice.
// Answers in DecisionResponse use the same ids.
type Questions map[string]Question

// Question is a single typed question evaluated against the state.
type Question struct {
	Type         string `json:"type"`               // "choice" | "score" | "noul"
	Instructions any    `json:"instructions"`       // string, object or array
	Criteria     any    `json:"criteria,omitempty"` // choice: options, score: ordered levels, noul: optional clarification
}

// Option is a named option of a Choice question.
type Option struct {
	Name        string
	Description string
}

// choiceCriteria keeps options in the declared order when serialized as a JSON object.
type choiceCriteria []Option

func (c choiceCriteria) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	buf.WriteByte('{')
	for i, o := range c {
		if i > 0 {
			buf.WriteByte(',')
		}
		k, err := json.Marshal(o.Name)
		if err != nil {
			return nil, err
		}
		v, err := json.Marshal(o.Description)
		if err != nil {
			return nil, err
		}
		buf.Write(k)
		buf.WriteByte(':')
		buf.Write(v)
	}
	buf.WriteByte('}')
	return buf.Bytes(), nil
}

// Choice creates a question that picks one of the given options.
func Choice(instructions string, options ...Option) Question {
	return Question{Type: QuestionChoice, Instructions: instructions, Criteria: choiceCriteria(options)}
}

// Score creates a question that scores the state on an ordered list of levels.
func Score(instructions string, levels ...string) Question {
	return Question{Type: QuestionScore, Instructions: instructions, Criteria: levels}
}

// Noul creates a question that estimates the probability that a statement is true.
func Noul(instructions string) Question {
	return Question{Type: QuestionNoul, Instructions: instructions}
}

// WithCriteria returns a copy of the question with custom criteria.
// Useful for Noul questions, to clarify what yes and no mean.
func (q Question) WithCriteria(criteria any) Question {
	q.Criteria = criteria
	return q
}

// DecisionRequest is the request for a decision call.
type DecisionRequest struct {
	State     any       `json:"state"`
	Model     string    `json:"model"`
	Questions Questions `json:"questions"`
}

// Answer is the typed answer to a single question.
type Answer struct {
	Type          string             `json:"type"`
	Choice        string             `json:"choice,omitempty"`
	Score         float64            `json:"score"`
	Noul          float64            `json:"noul"`
	Confidence    *float64           `json:"confidence,omitempty"` // nil for noul
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Legend        map[string]string  `json:"legend,omitempty"`
}

// DecisionResponse holds answers keyed by question id.
type DecisionResponse struct {
	Model    string            `json:"model"`
	Answers  map[string]Answer `json:"answers"`
	Metadata Metadata          `json:"metadata,omitempty"`
}

// Choice returns the selected option and its confidence for a choice question.
func (r *DecisionResponse) Choice(id string) (string, float64) {
	a := r.Answers[id]
	conf := 0.0
	if a.Confidence != nil {
		conf = *a.Confidence
	}
	return a.Choice, conf
}

// Score returns the score value for a score question.
func (r *DecisionResponse) Score(id string) float64 {
	return r.Answers[id].Score
}

// Noul returns the probability of "yes" for a noul question.
func (r *DecisionResponse) Noul(id string) float64 {
	return r.Answers[id].Noul
}

// decider is an optional provider capability for decision calls.
type decider interface {
	decide(ctx context.Context, req DecisionRequest, cfg CallConfig) (*DecisionResponse, error)
}

func validateQuestions(questions Questions) error {
	if len(questions) == 0 {
		return fmt.Errorf("no questions provided")
	}
	for id, q := range questions {
		if id == "" {
			return fmt.Errorf("question id must not be empty")
		}
		switch q.Type {
		case QuestionChoice, QuestionScore, QuestionNoul:
		default:
			return fmt.Errorf("question %q: unknown type %q", id, q.Type)
		}
		if q.Instructions == nil || q.Instructions == "" {
			return fmt.Errorf("question %q: instructions are required", id)
		}
		if (q.Type == QuestionChoice || q.Type == QuestionScore) && q.Criteria == nil {
			return fmt.Errorf("question %q: criteria are required for %s questions", id, q.Type)
		}
	}
	return nil
}

// Decide implements the Client interface
func (c *CommonClient) Decide(ctx context.Context, state any, questions Questions, opts ...CallOption) (*DecisionResponse, error) {
	if err := validateQuestions(questions); err != nil {
		return nil, err
	}

	p, cfg, err := c.prepareCall(opts...)
	if err != nil {
		return nil, err
	}

	d, ok := p.(decider)
	if !ok {
		return nil, fmt.Errorf("decisions are not supported by model %q", cfg.Model)
	}

	return d.decide(ctx, DecisionRequest{State: state, Model: cfg.Model, Questions: questions}, cfg)
}
