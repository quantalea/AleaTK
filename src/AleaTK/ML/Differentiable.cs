using System.Collections.Generic;

namespace AleaTK.ML
{
    public abstract class Differentiable : Symbol
    {
        private readonly List<Variable> _inputs = new List<Variable>();
        private readonly List<Variable> _outputs = new List<Variable>();
        private readonly List<Variable> _auxvars = new List<Variable>(); 

        public IEnumerable<Variable> Inputs => _inputs;

        public IEnumerable<Variable> Outputs => _outputs;

        public IEnumerable<Variable> AuxVars => _auxvars; 

        protected void AddInput(Variable input)
        {
            _inputs.Add(input);
        }

        protected void AddOutput(Variable output)
        {
            output.Owner = this;
            _outputs.Add(output);
        }

        protected void AddAuxVar(Variable auxvar)
        {
            _auxvars.Add(auxvar);
        }

        public abstract void Forward(Executor executor);

        public abstract void Backward(Executor executor);

        public virtual void Initialize(Executor executor)
        {
        }
    }

    public interface ILayer<T> {
        Variable<T> Output { get; }
    }
}